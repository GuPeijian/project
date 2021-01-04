import logging
import math
import os
import warnings

from typing import Optional, Tuple, Dict, List
import copy

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss

from src.transformers.activations import gelu, gelu_new, swish
from src.transformers.configuration_bert import BertConfig

from src.transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,

)
from src.transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm


#包含cache的bert

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

#self-attention
class SelfAttention(nn.Module):

    def __init__(self,config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.cache_key="self"

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_decoder_attention=False,
            encoder_hidden_states=None,
            key_padding_mask=None,
            layer_state : Optional[Dict[str, Optional[Tensor]]]=None,
            output_attentions=False,
    ):
        self.cache_key="encoder_decoder" if encoder_decoder_attention else "self"
        bsz, tgt_len, embed_dim = hidden_states.size()
        #load cache
        if layer_state is not None:
            saved_state=layer_state.get(self.cache_key,{})
            if "prev_key" in saved_state and encoder_decoder_attention:
                # previous time steps are cached - no need to recompute key and value if they are static
                encoder_hidden_states = None
        else:
            saved_state=None
            layer_state={}

        query_layer = self.query(hidden_states)

        if encoder_decoder_attention:
            if encoder_hidden_states is None:
                key_layer = value_layer = None
            else:
                key_layer=self.key(encoder_hidden_states)
                value_layer=self.value(encoder_hidden_states)
        else:
            key_layer=self.key(hidden_states)
            value_layer=self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        if key_layer is not None:
            key_layer = self.transpose_for_scores(key_layer)
        if value_layer is not None:
            value_layer = self.transpose_for_scores(value_layer)

        if saved_state is not None:
            key_layer,value_layer,key_padding_mask=self._use_saved_state(key_layer, value_layer,
                                        saved_state, key_padding_mask, encoder_decoder_attention, bsz)

        layer_state[self.cache_key]={
            "prev_key":key_layer,
            "prev_value":value_layer,
            "prev_key_padding_attention":key_padding_mask if not encoder_decoder_attention else None
        }

        assert key_layer is not None
        src_len=key_layer.size(2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores=attention_scores+attention_mask
        #若是cross attention 则 key_padding_mask是b*1*1*len,
        #若是self-attention 则 key_padding_mask 对于encoder和生成时的decoder是None
        #对训练时的decoder就是正常的b*len
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if not encoder_decoder_attention:
            assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

            if key_padding_mask is not None:  # don't attend to padding symbols
                attention_scores = attention_scores.view(bsz, self.num_attention_heads, tgt_len, src_len)
                reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
                extended_padding_mask = (1.0 - reshaped) * -10000.0

                attention_scores = attention_scores + extended_padding_mask
        else:
            assert key_padding_mask.size()[0]==bsz and key_padding_mask.size()[3]==src_len
            attention_scores = attention_scores.view(bsz, self.num_attention_heads, tgt_len, src_len)
            extended_padding_mask = (1.0 - key_padding_mask) * -10000.0
            attention_scores=attention_scores+extended_padding_mask




        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs=(context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


    def _use_saved_state(self, key_layer, value_layer, saved_state,
                         key_padding_mask, encoder_decoder_attention, bsz):
        # saved states are stored with shape (bsz, num_attention_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            prev_key=saved_state["prev_key"]
            assert prev_key is not None
            if encoder_decoder_attention:
                key_layer=prev_key
            else:
                assert key_layer is not None
                key_layer=torch.cat([prev_key,key_layer],dim=2)

        if "prev_value" in saved_state:
            prev_value=saved_state["prev_value"]
            assert prev_value is not None
            if encoder_decoder_attention:
                value_layer=prev_value
            else:
                assert value_layer is not None
                value_layer=torch.cat([prev_value,value_layer],dim=2)

        assert key_layer is not None and value_layer is not None

        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask",None)
        if prev_key_padding_mask is not None:
            if encoder_decoder_attention:
                new_key_padding_mask=prev_key_padding_mask
            else:
                new_key_padding_mask=torch.cat([prev_key_padding_mask,key_padding_mask],dim=1)
        else:
            new_key_padding_mask=key_padding_mask

        return key_layer,value_layer,new_key_padding_mask

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_decoder_attention=False,
            encoder_hidden_states=None,
            key_padding_mask=None,
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask,encoder_decoder_attention, encoder_hidden_states,
            key_padding_mask,layer_state,output_attentions,)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention=BertAttention(config)
        self.is_decoder=config.is_decoder
        if self.is_decoder:
            self.crossattention=BertAttention(config)
        self.intermediate=BertIntermediate(config)
        self.output=BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                key_padding_mask=None,
                layer_state = None,
                output_attentions=False,
                ):

        #key_padding_mask 属于{0,1},attention mask在encoder中即关注非pad的token，此时key_padding_mask可为None
        #而decoder中attention_mask为causal_mask，key_padding_mask为pad
        if self.is_decoder:
            if layer_state is None:
                layer_state={}
        self_attention_outputs=self.attention(
            hidden_states,
            attention_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            output_attentions = output_attentions,
            layer_state=layer_state,
        )
        attention_output=self_attention_outputs[0]
        outputs=self_attention_outputs[1:]  # add self attentions if we output attention weigh

        if self.is_decoder:
            if layer_state is None:
                layer_state={}

            cross_attention_outputs=self.crossattention(
                hidden_states=attention_output,
                encoder_decoder_attention=True,
                encoder_hidden_states=encoder_hidden_states,
                key_padding_mask=encoder_attention_mask,
                layer_state = layer_state,
            )
            attention_output=cross_attention_outputs[0]
            outputs=outputs+cross_attention_outputs[1:]

        intermediate_output=self.intermediate(attention_output)
        layer_output=self.output(intermediate_output,attention_output)
        outputs=(layer_output,)+outputs
        return outputs,layer_state


#作为encoder,attention_mask就是唯一的那个的attention_mask，其余的都为None
#作为decoder,self_attention中attention_mask为casual_mask,key_padding_mask为对应的pad_mask
#cross_attention中，只需要encoder_attention_mask作为key_padding_mask

class BertEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                key_padding_mask=None,
                decoder_past_key_values=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=False,
                ):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_decoder_cache = []

        for i,layer_module in enumerate(self.layer):
            layer_state = decoder_past_key_values[i] if decoder_past_key_values is not None else None
            if use_cache:
                hidden_states = hidden_states[:, -1:]
            layer_outputs,layer_past = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                key_padding_mask,
                layer_state,
                output_attentions,
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache.append(layer_past.copy())
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            if output_hidden_states:
                all_hidden_states=all_hidden_states+(hidden_states,)
        if use_cache:
            next_cache = ((encoder_hidden_states, encoder_attention_mask), next_decoder_cache)
        else:
            next_cache = None
        if not self.config.is_decoder:
            return BaseModelOutput(
                last_hidden_state=hidden_states,hidden_states=all_hidden_states,attentions=all_attentions
            )
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,past_key_values=next_cache,hidden_states=all_hidden_states,
                attentions=all_attentions
            )

class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """

    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        decoder_past_key_values=None,
        decoder_padding_mask=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if input_ids is None:
            raise ValueError("You must input ids")
        input_shape = input_ids.size()
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if self.config.is_decoder:
            if not use_cache:
                if decoder_padding_mask is None:
                    # 即padding的mask,不是causal mask
                    decoder_padding_mask = torch.ones(input_shape, device=device)
                    decoder_padding_mask=self.invert_attention_mask(decoder_padding_mask)

        #因为用了masked_fill，所以要invert
        if encoder_attention_mask is not None:
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            key_padding_mask=decoder_padding_mask,
            decoder_past_key_values=decoder_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not self.config.is_decoder:
            return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            )
        else:
            return BaseModelOutputWithPast(
                last_hidden_state=sequence_output,
                past_key_values=encoder_outputs.past_key_values,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions
            )

def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

class BertSeq2SeqModel(BertPreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        encoder_config=copy.copy(config)
        encoder_config.is_decoder=False
        self.encoder=BertModel(encoder_config)
        config.is_decoder=True
        self.decoder=BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        config.is_encoder_decoder=True
        self.config=config
        self.init_weights()

    def tie_weights(self):
        # for now no weights tying in encoder-decoder
        pass

    def get_decoder(self):
        return self.decoder


    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()


    def get_encoder(self):
        return self.encoder

    def get_output_embeddings(self):
        return self.cls.predictions.decoder


    def forward(self,
                input_ids,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                decoder_input_ids=None,
                encoder_outputs:Optional[Tuple]=None,
                decoder_padding_mask=None,
                decoder_past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                labels=None,
                ):

        if decoder_input_ids is None:
            use_cache=False

        if labels is not None:
            use_cache = False


        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if encoder_outputs is None:
            encoder_outputs=self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        if not isinstance(encoder_outputs,BaseModelOutputWithPooling):
            encoder_outputs=BaseModelOutputWithPooling(
                last_hidden_state=encoder_outputs[0],
                pooler_output=encoder_outputs[1],
                hidden_states=encoder_outputs[2] if len(encoder_outputs)>2 else None,
                attentions=encoder_outputs[3] if len(encoder_outputs)>3 else None,
            )


        encoder_hidden_states=encoder_outputs[0]
        if decoder_padding_mask is not None:
            if decoder_padding_mask.type() == 'torch.LongTensor':
                decoder_padding_mask=decoder_padding_mask.float()


        decoder_outputs=self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=None,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            decoder_past_key_values=decoder_past_key_values,
            decoder_padding_mask=decoder_padding_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            )

        decoder_hidden_states=decoder_outputs[0]
        prediction_scores = self.cls(decoder_hidden_states)

        lm_loss=None
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.reshape(-1))

        return Seq2SeqLMOutput(
            loss=lm_loss,
            logits=prediction_scores,
            decoder_past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(self, decoder_input_ids,past,attention_mask,use_cache,**kwargs):
        assert past is not None

        encoder_outputs,decoder_past_key_values=past

        return{
            "input_ids":None,
            "encoder_outputs": encoder_outputs,
            "decoder_past_key_values":decoder_past_key_values,
            "decoder_input_ids":decoder_input_ids,
            "attention_mask":attention_mask,
            "use_cache":use_cache,
        }

    def _force_token_ids_generation(self, scores, token_ids) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0"""
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(logits, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_past_key_values) = past
        reordered_past = []
        for layer_past in decoder_past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        new_enc_out = enc_out if enc_out is None else enc_out.index_select(0, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past












