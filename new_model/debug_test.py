import sys
sys.path.append("..")
import torch
from UniLM import FuseModel

model_list=[]

model_path1="./weibo_output_no_bow"
model_path2="./sentiment_output_bow_clean"
device = torch.device("cuda")

model1=FuseModel.from_pretrained(model_path1)
model1.to(device)
model_list.append(model1)
model2=FuseModel.from_pretrained(model_path2)
model2.to(device)
model_list.append(model2)


print("success")