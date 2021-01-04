import torch
import copy
from src.transformers.configuration_bert import BertConfig
from src.transformers  import BertTokenizer

from new_model.UniLM import FuseModel

a=[[101,202,303,404,102,0,0],[101,202,303,404,102,0,0]]
b=[[1,1,1,1,1,0,0],[1,1,1,1,1,0,0]]

input_ids=torch.tensor(a)
attention_mask=torch.tensor(b)


#config = BertConfig.from_pretrained('E:\\bert')


model=FuseModel.from_pretrained('E:\\bert')
#print(model.embeddings.word_embeddings.weight.t().size())

#tokenizer = BertTokenizer.from_pretrained('E:\\bert')

a=[[101,230,990,5625,220,102,0,0,0,202,660,102,0,0]]
attention_mask=[[1,1,1,1,1,1,0,0,0,1,1,1,0,0]]
token_type_ids=[[0,0,0,0,0,0,0,0,0,1,1,1,1,1]]
label=[[202,505,102,0,0]]
sentiment=[[0]]

input_ids=torch.tensor(a)
attention_mask=torch.tensor(attention_mask)
token_type_ids=torch.tensor(token_type_ids)
label=torch.tensor(label)
sentiment=torch.tensor(sentiment)

b=model(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,labels=label,sentiment_input=sentiment)

print(b[0])
"""
a=model.generate(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,max_length=30)

print(a)
"""


