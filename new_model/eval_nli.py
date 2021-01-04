import sys

sys.path.append("..")

import torch

import os, json, logging, random
from tqdm import  tqdm
import numpy as np
import argparse

from transformers import BertForSequenceClassification

from torch.utils.data import Dataset, DataLoader,SequentialSampler


class tokendataset(Dataset):
    def __init__(self, list_file):
        self.dataset = list_file

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return (data)


def padding_nli_fn(datalist):
    input_ids=[]
    attention_masks=[]
    token_type_ids=[]
    labels=[]
    length=[]
    #concatenate
    for item in datalist:
        context=item["context"]
        comment=item["comment"]
        input_id=context+comment
        attention_mask=[1]*len(input_id)
        token_type_id=[0]*len(context)+[1]*len(comment)

        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        labels.append(item["label"])
        length.append(len(input_id))
    #pad
    max_len=max(length)
    for i in range(len(input_ids)):
        pad_len=max_len-len(input_ids[i])
        input_ids[i]=input_ids[i]+[0]*pad_len
        attention_masks[i]=attention_masks[i]+[0]*pad_len
        token_type_ids[i]=token_type_ids[i]+[1]*pad_len


    batch={}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["token_type_ids"]=token_type_ids
    batch["labels"] = labels
    return batch


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def main():
    parser = argparse.ArgumentParser()
    # data and model
    parser.add_argument("--data_path", default='/home/gpj/project/all_data/encode_data/nli_data/test_nli_data.json', type=str, )
    parser.add_argument("--model_name_or_path", default="/home/gpj/project/all_model/nli", type=str, )
    parser.add_argument("--batch_size", default=16, type=int, )
    parser.add_argument("--seed", type=int, default=66, help="初始化随机种子")

    args = parser.parse_args()

    device = torch.device("cuda")
    # Set seed
    set_seed(args)

    # load json data
    with open(args.data_path, 'r')as f:
        data_file = json.load(f)

    train_dataset = tokendataset(data_file)
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, collate_fn=padding_nli_fn, sampler=train_sampler)

    model = BertForSequenceClassification.from_pretrained(args.model_name_or_path)

    model.to(device)

    model.eval()

    logger.info("start_training")

    for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
        input_ids, attention_mask,token_type_ids,labels = \
            batch["input_ids"],batch["attention_mask"],batch["token_type_ids"],batch["labels"]

        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)
        token_type_ids=torch.tensor(token_type_ids).to(device)
        labels = torch.tensor(labels).to(device)

        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None,)

        logits=outputs[0]
        b=logits.size()
        c=b





if __name__ == '__main__':
    main()





