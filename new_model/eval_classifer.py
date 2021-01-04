# -*- coding:utf-8 -*-
import torch
from transformers import BertForSequenceClassification
import json
from transformers import BertTokenizer
tokenizer_path = "/home/gpj/project/new_model/sentiment_classifer/"
model_path = "/home/gpj/project/new_model/sentiment_classifer1/"

tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

input_text1="死就死了，伤就伤了，这是多么轻又重的话"

input_text2="自我安慰她是要去找她的孩子的她放心不下那四个小家伙吧"

input_ids1= torch.tensor([tokenizer.encode(input_text1, max_length=50, truncation=True,add_special_tokens=True)])
input_ids2= torch.tensor([tokenizer.encode(input_text2, max_length=50, truncation=True,add_special_tokens=True)])

model = BertForSequenceClassification.from_pretrained(model_path,num_labels=3)

device= torch.device("cuda")

model.to(device)

model.eval()

input_ids1=input_ids1.to(device)


input_ids2=input_ids2.to(device)

output1=model(input_ids=input_ids1,labels=torch.tensor([0]).to(device))

output2=model(input_ids=input_ids2,labels=torch.tensor([1]).to(device))

print(output1)
print(output2)

"""
optimizer = torch.optim.Adam(model.parameters(),lr=3e-5,eps=1e-6)
model.train()
model.zero_grad()

a,b=model.rl_generate(input_ids=input_ids,topp=0.96,no_repeat_ngram_size=3,max_length=290,min_length=257)

#c=tokenizer.decode(a[0][input_ids.size()[1]:])
reward=[0.1,0.2]
reward=r=torch.tensor(reward)
print(b)
L=b.mul(reward).sum()
L.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度截断
optimizer.step()
"""





