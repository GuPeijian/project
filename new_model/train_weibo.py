import sys

sys.path.append("..")
from UniLM import FuseModel

import torch
from torch.utils.tensorboard import SummaryWriter

import os, json, logging, random
from tqdm import trange, tqdm
import numpy as np
import argparse

from transformers import AdamW, BertTokenizer
from transformers import get_linear_schedule_with_warmup
import math

from torch.utils.data import Dataset, DataLoader


class tokendataset(Dataset):
    def __init__(self, data_file):
        self.file_path = data_file
        dataset = []
        file_raws = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                file_raws += 1
                dataset.append(_)
        self.file_raws = file_raws
        self.dataset = dataset

    def __len__(self):
        return self.file_raws

    def __getitem__(self, idx):
        data = json.loads(self.dataset[idx][:-1]) #remove /n
        return data


def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    labels = []
    sentiment = []
    context_length = []
    comment_length = []
    for item in data_list:
        context_length.append(len(item["context"]))
        comment_length.append(len(item["comment"]))
        #sentiment.append([item["sentiment"]])
    max_context_len = max(context_length)
    max_comment_len = max(comment_length)
    for i, item in enumerate(data_list):
        context_pad_len = max_context_len - context_length[i]
        comment_pad_len = max_comment_len - comment_length[i]
        context = item["context"]
        comment = item["comment"]
        # pad context ，此处似乎无法调用tokenizer，只能人为设定PAD为0
        attention_mask = [1] * context_length[i]
        context.extend([0] * context_pad_len)
        attention_mask.extend([0] * context_pad_len)
        token_type_id = [0] * max_context_len
        # pad comment
        attention_mask.extend([1] * comment_length[i])
        comment.extend([0] * comment_pad_len)
        attention_mask.extend([0] * comment_pad_len)
        token_type_id.extend([1] * max_comment_len)
        # combine
        input_ids.append(context + comment)
        attention_masks.append(attention_mask)
        token_type_ids.append(token_type_id)
        labels.append(comment)
    # batch
    batch = {}

    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["token_type_ids"] = token_type_ids
    batch["labels"] = labels
    #batch["sentiment"] = sentiment
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
    parser.add_argument("--data_path", default="/home/gpj/project/all_data/all/all_data.txt", type=str, )
    parser.add_argument("--model_name_or_path", default='/home/gpj/bert-chinese', type=str, )
    parser.add_argument("--output_dir", default="/home/gpj/project/all_model/all", type=str, )

    parser.add_argument("--batch_size", default=10, type=int, )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, )
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="学习率衰减")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="梯度裁减值")
    parser.add_argument("--num_train_epochs", default=2, type=int, help="训练epochs次数", )
    parser.add_argument("--warmup_rate", default=0.1, type=int, help="学习率线性预热步数")
    parser.add_argument("--logging_steps", type=int, default=100, help="每多少步打印日志")
    parser.add_argument("--log_dir", default="/home/gpj/project/log/all", type=str, )
    parser.add_argument("--seed", type=int, default=100, help="初始化随机种子")
    parser.add_argument("--max_steps", default=1000000, type=int, help="训练的总步数", )
    parser.add_argument("--save_steps", default=40000, type=int, help="保存的间隔steps", )
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--gpus', default=6, type=int, help='number of gpus per node')
    parser.add_argument('--nodes', default=1, type=int, metavar='N')
    parser.add_argument("--overwrite_output_dir", default=False, action="store_true", help="是否覆盖输出文件夹")

    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '6666'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,5,6,7'
    args.world_size = args.gpus * args.nodes

    if (
            os.path.exists(args.output_dir) and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir)
        )
    torch.distributed.init_process_group('nccl', init_method='env://')
    device = torch.device("cuda:{}".format(args.local_rank))

    # Setup logging
    if args.local_rank == 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

    # Set seed
    set_seed(args)


    train_dataset = tokendataset(args.data_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, num_workers=0, batch_size=args.batch_size, shuffle=False,
                                  drop_last=True, collate_fn=padding_fuse_fn, sampler=train_sampler)

    model = FuseModel.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    SPECIAL_TOKENS_DICT = {'additional_special_tokens': ["<user1>", "<user2>"]}
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                      output_device=args.local_rank, find_unused_parameters=True)
    t_total = args.max_steps
    if args.local_rank == 0:
        tb_writer = SummaryWriter(args.log_dir)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    num_train_steps = math.floor((train_dataset.__len__()) / (
                args.batch_size * args.gpus * args.gradient_accumulation_steps)) * args.num_train_epochs
    num_warmup_steps = math.floor(num_train_steps * args.warmup_rate)
    #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.train()
    model.zero_grad()
    if args.local_rank == 0:
        logger.info("start_training")

    #unique_tokens_file = "./unique_tokens.json"
    #with open(unique_tokens_file, 'r', encoding='utf-8') as f:
        #unique_tokens = json.load(f)
    # assert len(unique_tokens[0]) == len(tokenizer)
    # unique_tokens = torch.tensor(unique_tokens, device=device)

    for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, attention_mask, token_type_ids, labels = \
                batch["input_ids"], batch["attention_mask"], batch["token_type_ids"], batch["labels"]

            # 考虑label掐头，input_ids，attention_mask,token_type_ids要去尾,
            input_ids = torch.tensor(input_ids).to(device)
            attention_mask = torch.tensor(attention_mask).to(device)
            token_type_ids = torch.tensor(token_type_ids).to(device)
            labels = torch.tensor(labels).to(device)[:, 1:]
            #sentiment = torch.tensor(sentiment).to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels,)
            loss = outputs[0]
            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度截断
                optimizer.step()
                #scheduler.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank == 0:
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        logs = {}
                        loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                        logs["loss"] = loss_scalar
                        logging_loss = tr_loss

                        for key, value in logs.items():
                            tb_writer.add_scalar(key, value, global_step)
                        print(json.dumps({**logs, **{"step": global_step}}))

                if args.save_steps > 0 and global_step % args.save_steps == 0 and torch.distributed.get_rank() == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (model.module if hasattr(model, "module") else model)
                    model_to_save.save_pretrained(output_dir)

            if args.max_steps > 0 and global_step > t_total:
                break
        if args.max_steps > 0 and global_step > t_total:
            break

    tb_writer.close()
    if args.local_rank == 0:
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss / global_step)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if torch.distributed.get_rank() == 0:
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == '__main__':
    main()





