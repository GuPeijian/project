import logging
import math
import os
import warnings
from typing import Optional, Tuple, Dict, List

import torch
import torch.utils.checkpoint
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss
import sys
sys.path.append("..")

from src.transformers.activations import gelu, gelu_new, swish
from src.transformers.configuration_bert import BertConfig

from src.transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
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

#包含cache的UniLM
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

        self.cache_key = "self"
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            layer_state : Optional[Dict[str, Optional[Tensor]]]=None,
            output_attentions=False,
    ):
        bsz, tgt_len, embed_dim = hidden_states.size()
        #load cache
        if layer_state is not None:
            saved_state=layer_state.get(self.cache_key,{})
        else:
            saved_state=None
            layer_state={}

        query_layer = self.query(hidden_states)

        #对于decode时候，输入仅为len=1的
        key_layer=self.key(hidden_states)
        value_layer=self.value(hidden_states)

        query_layer = self.transpose_for_scores(query_layer)
        if key_layer is not None:
            key_layer = self.transpose_for_scores(key_layer)
        if value_layer is not None:
            value_layer = self.transpose_for_scores(value_layer)

        if saved_state is not None:
            key_layer,value_layer=self._use_saved_state(key_layer, value_layer,saved_state)

        layer_state[self.cache_key]={
            "prev_key":key_layer,
            "prev_value":value_layer
        }

        assert key_layer is not None
        src_len=key_layer.size(2)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores=attention_scores+attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs=(context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

    def _use_saved_state(self, key_layer, value_layer, saved_state,):
        # saved states are stored with shape (bsz, num_attention_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            prev_key=saved_state["prev_key"]
            assert prev_key is not None

            assert key_layer is not None
            key_layer=torch.cat([prev_key,key_layer],dim=2)

        if "prev_value" in saved_state:
            prev_value=saved_state["prev_value"]
            assert prev_value is not None

            assert value_layer is not None
            value_layer=torch.cat([prev_value,value_layer],dim=2)

        assert key_layer is not None and value_layer is not None

        return key_layer,value_layer

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
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states, attention_mask,layer_state,output_attentions,)
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
        self.intermediate=BertIntermediate(config)
        self.output=BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask=None,
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
            output_attentions = output_attentions,
            layer_state=layer_state,
        )
        attention_output=self_attention_outputs[0]
        outputs=self_attention_outputs[1:]  # add self attentions if we output attention weigh

        intermediate_output=self.intermediate(attention_output)
        layer_output=self.output(intermediate_output,attention_output)
        outputs=(layer_output,)+outputs
        return outputs,layer_state

class BertEncoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask=None,
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
                if layer_state is not None:
                    hidden_states = hidden_states[:, -1:]
            layer_outputs,layer_past = layer_module(
                hidden_states,
                attention_mask,
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
            next_cache = (next_decoder_cache)
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

def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache

class UniLMModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        decoder_past_key_values=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        labels=None,
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
        #return [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask: torch.Tensor =  attention_mask[:, None, None, :]

        unilm_mask=self.compute_attention_mask(token_type_ids)

        extended_attention_mask=extended_attention_mask+unilm_mask

        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        #因为使用了两个mask相加，所以需要2
        extended_attention_mask = (2.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            decoder_past_key_values=decoder_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        prediction_scores = self.cls(sequence_output)
        lm_loss=None
        input_len=(1-token_type_ids[0]).sum()
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            prediction_scores=prediction_scores[:,(input_len-1):-1,:].contiguous()
            lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.reshape(-1))

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=prediction_scores,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


    #计算unilm中的那种mask
    def compute_attention_mask(self,token_type_ids):
        """计算seq2seq的mask矩阵"""
        device = token_type_ids.device
        seq_len = token_type_ids.shape[1]
        # 1,num_heads,seq,seq
        ones = torch.ones(1,self.config.num_attention_heads,seq_len,seq_len,device=device)

        # 下三角矩阵
        a_mask = torch.tril(ones)
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(1)
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3)

        # batch,num_heads,seq,seq
        a_mask = (1-s_ex13) * (1-s_ex12) + a_mask * s_ex13

        # a_mask = a_mask.view(-1,seq_len,seq_len).contiguous()
        return a_mask

    def prepare_inputs_for_generation(self, input_ids,past,attention_mask,token_type_ids,use_cache,**kwargs):
        if past is not None:
            decoder_past_key_values=past
        else:
            decoder_past_key_values=None
        return{
            "input_ids":input_ids,
            "decoder_past_key_values":decoder_past_key_values,
            "attention_mask":attention_mask,
            "token_type_ids":token_type_ids,
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
        (decoder_past_key_values) = past
        reordered_past = []
        for layer_past in decoder_past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        past = (reordered_past)
        return past

class SentimentEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sentiment_embeddings = nn.Embedding(3, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sentiment_input=None):
        assert sentiment_input is not None
        embeddings=self.sentiment_embeddings(sentiment_input)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SoftEntropy(nn.Module):
   def __init__(self):
      super(SoftEntropy, self).__init__()
      self.logsoftmax = nn.LogSoftmax(dim=1)
      self.softmax = nn.Softmax(dim=1)

   def forward(self, inputs, targets):
      log_probs = self.logsoftmax(inputs)
      #loss = (- self.softmax(targets).detach() * log_probs).mean(0).sum()
      loss = (-self.normalize_target(targets) * log_probs).mean(0).sum()
      return loss

   def normalize_target(self,targets):
       target_lens=targets.sum(dim=1).unsqueeze(1).float() + 1e-6 #[B,1]   1e-6 incase len=0
       normalized_target=torch.div(targets,target_lens)
       return normalized_target

class FuseModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sentiment_embeddings=SentimentEmbedding(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.cls = BertOnlyMLMHead(config)
        self.bowloss=SoftEntropy()
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            decoder_past_key_values=None,
            use_cache=False,
            output_attentions=None,
            output_hidden_states=None,
            labels=None,
            sentiment_input=None,
            unique_tokens=None,
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

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        )
        if sentiment_input is not None:
            sentiment_output=self.sentiment_embeddings(sentiment_input=sentiment_input)
            embedding_output=torch.cat((sentiment_output,embedding_output),dim=1).contiguous()
            #here we need to add an extra attention mask for the sentiment
            #in PLATO its attention serves just like an context
            #so we can just add an extra mask on the beginning of the attention/toekn_type

            extra_attention_mask=torch.ones(input_shape[0],dtype=torch.long,device=device).unsqueeze(1)
            extra_token_type_id = torch.zeros(input_shape[0], dtype=torch.long, device=device).unsqueeze(1)
            attention_mask=torch.cat((extra_attention_mask,attention_mask),dim=1).contiguous()
            token_type_ids=torch.cat((extra_token_type_id,token_type_ids),dim=1).contiguous()

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # return [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask: torch.Tensor = attention_mask[:, None, None, :]
        unilm_mask = self.compute_attention_mask(token_type_ids)
        extended_attention_mask = extended_attention_mask + unilm_mask
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        # 因为使用了两个mask相加，所以需要2
        extended_attention_mask = (2.0 - extended_attention_mask) * -10000.0

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            decoder_past_key_values=decoder_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        prediction_scores = self.cls(sequence_output)
        loss=None
        lm_loss = None
        bow_loss=None
        input_len = (1 - token_type_ids[0]).sum()
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            decoder_scores = prediction_scores[:, input_len:-1, :].contiguous()
            lm_loss = loss_fct(decoder_scores.view(-1, self.config.vocab_size), labels.reshape(-1))
            if sentiment_input is not None:
                bow_socres=prediction_scores[:,0,:].squeeze(1)
                bow_target=torch.zeros(input_shape[0],self.config.vocab_size,device=device)
                bow_target.scatter_(1,labels,1)
                bow_mask=unique_tokens[sentiment_input.squeeze(1)]
                bow_target.mul(bow_mask)
                #bow_target[:,self.config.pad_token_id]=0
                bow_loss=self.bowloss(bow_socres,bow_target)
            if sentiment_input is not None:
                loss=lm_loss+bow_loss
            else:
                loss = lm_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=prediction_scores,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    # 计算unilm中的那种mask
    def compute_attention_mask(self, token_type_ids):
        """计算seq2seq的mask矩阵"""
        device = token_type_ids.device
        seq_len = token_type_ids.shape[1]
        # 1,num_heads,seq,seq
        ones = torch.ones(1, self.config.num_attention_heads, seq_len, seq_len, device=device)

        # 下三角矩阵
        a_mask = torch.tril(ones)
        s_ex12 = token_type_ids.unsqueeze(1).unsqueeze(1)
        s_ex13 = token_type_ids.unsqueeze(1).unsqueeze(3)

        # batch,num_heads,seq,seq
        a_mask = (1 - s_ex13) * (1 - s_ex12) + a_mask * s_ex13

        # a_mask = a_mask.view(-1,seq_len,seq_len).contiguous()
        return a_mask

    def prepare_inputs_for_generation(self, input_ids, past, attention_mask, token_type_ids, use_cache, **kwargs):
        if past is not None:
            decoder_past_key_values = past
        else:
            decoder_past_key_values = None
        sentiment=kwargs["sentiment_input"]
        return {
            "input_ids": input_ids,
            "decoder_past_key_values": decoder_past_key_values,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "use_cache": use_cache,
            "sentiment_input": sentiment,
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
        (decoder_past_key_values) = past
        reordered_past = []
        for layer_past in decoder_past_key_values:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            reordered_past.append(layer_past_new)

        past = (reordered_past)
        return past
