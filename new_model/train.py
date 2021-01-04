
from new_model.my_model import BertSeq2SeqModel
from src.transformers.configuration_bert import BertConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

import os,json,codecs,logging,random
from tqdm import trange,tqdm
import numpy as np
import argparse

from torch.utils.data import Dataset, DataLoader

class tokendataset(Dataset):
    def __init__(self,list_file):
        self.dataset=list_file

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data=self.dataset[idx]
        return(data)

def padding_fn(data_list):
    length=[]
    for item in data_list:
        length.append(len(item))
    pad_len=max(length)
    for item in data_list:
        if len(item)<pad_len:
            item.extend([0]*(pad_len-len(item)))
    return data_list

def padding_all_fn(data_list):
    input_ids=[]
    input_attention_mask=[]
    labels=[]
    decoder_mask=[]
    for item in data_list:
        input_ids.append(item["input_ids"])
        input_attention_mask.append(item["input_attention_mask"])
        labels.append(item["labels"])
        decoder_mask.append(item["decoder_mask"])
    input_ids=padding_fn(input_ids)
    input_attention_mask=padding_fn(input_attention_mask)
    labels=padding_fn(labels)
    decoder_mask=padding_fn(decoder_mask)
    batch={}
    batch["input_ids"]=input_ids
    batch["input_attention_mask"]=input_attention_mask
    batch["labels"]=labels
    batch["decoder_mask"]=decoder_mask
    return batch

def padding_unilm_fn(data_list):
    input_ids=[]
    attention_mask=[]
    token_type_ids=[]
    labels=[]
    length=[]
    for item in data_list:
        input_ids.append(item["input_ids"])
        attention_mask.append(item["attention_mask"])
        token_type_ids.append(item["token_type_ids"])
        labels.append(item["labels"])
        length.append(len(item["input_ids"]))
    max_len=max(length)
    for i ,_ in enumerate(input_ids):
        pad_len=max_len-len(input_ids[i])
        input_ids[i].extend([0]*pad_len)
        attention_mask[i].extend([0]*pad_len)
        token_type_ids[i].extend([1]*pad_len)
        labels[i].extend([0]*pad_len)
    batch={}
    batch["input_ids"]=input_ids
    batch["attention_mask"]=attention_mask
    batch["token_type_ids"]=token_type_ids
    batch["labels"]=labels
    return batch




logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()
    #data and model
    parser.add_argument("--data_path",default='E:\\tokenized_ids.json',type=str,required=False,)
    parser.add_argument("--model_name_or_path",default='E:\\bert',type=str,required=False,)
    parser.add_argument("--output_dir",default='./model_output',type=str,required=False,)

    parser.add_argument("--batch_size", default=4, type=int,)
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,)
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,)
    parser.add_argument("--do_train",default=True, action="store_true", help="是否fine tuning")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="训练epochs次数",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=500, help="每多少步打印日志")
    parser.add_argument("--seed", type=int, default=42, help="初始化随机种子")
    parser.add_argument("--max_steps",default=200000,type=int,help="训练的总步数",)
    parser.add_argument("--save_steps", default=20000, type=int, help="保存的间隔steps", )
    parser.add_argument("--overwrite_output_dir",default=False, action="store_true", help="是否覆盖输出文件夹")
    args=parser.parse_args()

    if (
        os.path.exists(args.output_dir) and os.listdir(args.output_dir)
        and args.do_train and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(args)

    #load json data
    with open(args.data_path, 'r')as f:
        data_file = json.load(f)

    train_dataset = tokendataset(data_file)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=padding_all_fn)

    if args.do_train:
        model = BertSeq2SeqModel.from_pretrained('E:\\bert', double_model=True)
        model.to(device)
        t_total = args.max_steps
        optimizer = torch.optim.Adam(model.parameters(),lr=args.learning_rate,eps=args.adam_epsilon)
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        model.zero_grad()
        logger.info("start_training")

        for epoch in trange(int(args.num_train_epochs),desc='Epoch'):
            for step,batch in enumerate(tqdm(train_dataloader,desc='Iteration')):
                input_ids, input_attention_mask, labels, decoder_mask = \
                    batch["input_ids"], batch["input_attention_mask"], batch["labels"], batch["decoder_mask"]

                input_ids = torch.tensor(input_ids).to(device)
                input_attention_mask = torch.tensor(input_attention_mask).to(device)
                labels = torch.tensor(labels).to(device)
                decoder_mask = torch.tensor(decoder_mask).to(device)
                #preare for decode input
                decoder_input_ids=labels.clone()[:,:-1]
                labels=labels[:,1:]
                decoder_mask=decoder_mask[:,:-1]
                outputs=model(input_ids=input_ids,
                              attention_mask=input_attention_mask,
                              decoder_input_ids=decoder_input_ids,
                              decoder_padding_mask=decoder_mask,
                              labels=labels)
                loss=outputs[0]
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度截断
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                    if args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (model.module if hasattr(model, "module") else model)
                        model_to_save.save_pretrained(output_dir)

                if args.max_steps > 0 and global_step > t_total:
                    break
            if args.max_steps > 0 and global_step > t_total:
                break

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)

if __name__ =='__main__':
    main()





