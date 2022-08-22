# ================== #
# 1. Add Mutil-GPUs;
# ================== #

import os
import json
import torch
import random
import argparse
import numpy as np
import seqeval.metrics as se

from utils import *
from model import BertCRFForTokenClassification

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_cosine_schedule_with_warmup
)

import warnings
warnings.filterwarnings("ignore")

import logging
logger = logging.getLogger(__name__)


class myDataset(Dataset):
    
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def __len__(self):
        return len(self.features)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_mutil_gpu:
        torch.cuda.manual_seed_all(args.seed)


def get_labels(args):
    return open(os.path.join(args.data_path, args.label_name), "r").readline()


def load_examples(args, data_name, tokenizer, labels, pad_token_label_id, mode):
    cached_features_file = args.data_path + "/Cache/cached_{}_{}".format(data_name, mode)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", os.path.join(args.data_path, data_name))
        examples = read_examples_from_file(os.path.join(args.data_path, data_name), mode)
        features = convert_examples_to_features(
            examples, labels, tokenizer, pad_token_label_id=pad_token_label_id, max_seq_length=args.max_seq_length,
            sep_token_extra = False  # RoBERTa uses an extra separator b/w pairs of sentences
        )
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    return myDataset(features)


def myFn(batch, pad_token_label_id):
    input_ids = pad_sequence([feature.input_ids for feature in batch], batch_first=True)
    attention_mask = pad_sequence([feature.input_mask for feature in batch], batch_first=True)
    labels = pad_sequence([feature.label_ids for feature in batch], batch_first=True, padding_value=pad_token_label_id)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def train(args, train_loader, valid_loader, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    
    logger.info("********** Running training **********")
    logger.info("    Num Train Loader = %d", len(train_loader))
    logger.info("    Num Epochs = %d", args.num_train_epochs)
    logger.info("    Batch Size = %d", args.train_batch_size)
    
    # -------------- #
    #   差分学习率
    # -------------- #
    bert_params = list(map(id, model.bert.parameters()))
    base_params = filter(lambda p: id(p) not in bert_params, model.parameters())

    optimizer_grouped_parameters = [
        { "params": base_params, "lr": args.lr * 100, "weight_decay": 0.0 },
        { "params": model.bert.parameters(), "lr": args.lr, "weight_decay": 0.0 }
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, len(train_loader) * args.num_train_epochs)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    
    global_step = 0
    best_dev_f1 = -100.0
    for epoch in range(args.num_train_epochs):
        model.train()
        
        loss_list = list()
        train_bar = tqdm(train_loader)
        for batch in train_bar:
            optimizer.zero_grad()
            batch = { k: v.to(args.device) for k, v in batch.items()}
            
            loss = model(pad_token_label_id=pad_token_label_id, **batch)
            loss_list.append(loss.item())
            
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            
            if args.fp16:
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            train_bar.set_description("[Epoch {}/{} | Step {}] Loss: {:.5f}".format(
                epoch, args.num_train_epochs, global_step, sum(loss_list)/len(loss_list)
            ))
            
            if args.do_valid:
                global_step += 1

                if global_step % args.eva_step == 0:
                    results, _ = evaluate(args, valid_loader, model, tokenizer, labels, pad_token_label_id, mode="valid")

                    if (results['f1-micro'] + results['f1-macro']) / 2 > best_dev_f1:
                        # -------------- #
                        #     Saving
                        # -------------- #
                        best_dev_f1 = (results['f1-micro'] + results['f1-macro']) / 2
                        model_to_save = model.module if hasattr(model, "module") else model

                        model_save_path = args.output_path + "/" + args.save_name

                        model_to_save.save_pretrained(model_save_path)

                        logger.info("    Best epoch: %d", epoch)
                        logger.info("    Best f1: {}".format(best_dev_f1))
                        
        if not args.do_valid:
            # -------------- #
            #     Saving
            # -------------- #
            best_dev_f1 = (results['f1-micro'] + results['f1-macro']) / 2
            model_to_save = model.module if hasattr(model, "module") else model

            model_save_path = args.output_path + "/" + args.save_name

            model_to_save.save_pretrained(model_save_path)


def evaluate(args, valid_loader, model, tokenizer, label_list, pad_token_label_id, mode="valid"):
    """ Evaluation """
    
    logger.info("********** Running evaluation **********")
    logger.info("    Num examples = %d", len(valid_loader))
    logger.info("    Batch size = %d", args.valid_batch_size)
    
    model.eval()

    preds, out_label_ids = list(), list()
    valid_bar = tqdm(valid_loader)
    with torch.no_grad():
        for batch in valid_bar:
            batch = { k: v.to(args.device) for k, v in batch.items()}
            labels = batch.pop("labels")
        
            best_path = model(pad_token_label_id=pad_token_label_id, **batch)
            
            preds.extend(best_path.tolist())
            out_label_ids.extend(labels.tolist())
    
    label_map = { i: label for i, label in enumerate(label_list) }
    
    out_label_list = [[] for _ in range(len(out_label_ids))]
    preds_list = [[] for _ in range(len(out_label_ids))]
    
    for i in range(len(out_label_ids)):
        for j in range(len(out_label_ids[i])):
            if out_label_ids[i][j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])
      
    results = {
        "p-micro": se.precision_score(out_label_list, preds_list, average='micro'),
        "r-micro": se.recall_score(out_label_list, preds_list, average='micro'),
        "f1-micro": se.f1_score(out_label_list, preds_list, average='micro'),

        "p-marco": se.precision_score(out_label_list, preds_list, average='macro'),
        "r-marco": se.recall_score(out_label_list, preds_list, average='macro'),
        "f1-macro": se.f1_score(out_label_list, preds_list, average='macro'),
    }

    info = se.classification_report(out_label_list, preds_list, digits=4, output_dict=True)
    for key in info:
        value = info[key]
        for v in value:
            value[v] = float(value[v])

    with open(f"{args.output_path}/{args.save_name}/prf_report_dict.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    print(
        se.classification_report(out_label_list, preds_list, digits=4),
        file=open(f"{args.output_path}/{args.save_name}/prf_report_dict.json", "w")
    )
    
    logger.info("********** Eval results **********")
    for key in results.keys():
        logger.info("    %s = %s", key, str(results[key]))
    logger.info("    Avg F1 = %s" % ((results["f1-micro"] + results["f1-macro"]) / 2))

    return results, preds_list


def main():
    parser = argparse.ArgumentParser()
    
    # --------------------------------------------- #
    # 1.所有文件都是相对于根目录文件路径;
    # --------------------------------------------- #
    parser.add_argument("--root", default="/tf/FangGexiang/1.CAILED", type=str, help="根路径")
    parser.add_argument("--data_path", default="/tf/FangGexiang/1.CAILED/Data", type=str, help="存放数据的文件夹")
    parser.add_argument("--output_path", default="/tf/FangGexiang/1.CAILED/ModelSaved", type=str, help="模型存储位置")
    parser.add_argument("--save_name", type=str, default="baseline", help="Model save name.")
    parser.add_argument("--ptm_path", default="/tf/FangGexiang/3.SememeV2/pretrained_model", type=str, help="预训练模型位置")
    parser.add_argument("--result_path", default="/result_saved/", type=str, help="结果输出位置")
    
    # --------------------------------------------- #
    # 2.数据相关的参数;
    # --------------------------------------------- #
    parser.add_argument("--train_data_name", default="train.jsonl", type=str, help="训练数据文件名")
    parser.add_argument("--valid_data_name", default="valid.jsonl", type=str, help="验证数据文件名")
    parser.add_argument("--test_data_name", default="test_stage1.jsonl", type=str, help="测试数据文件名")
    parser.add_argument("--label_name", default="bio_label.txt", type=str, help="拢共有哪些Label")
    
    # --------------------------------------------- #
    # 3.模型相关的参数;
    # --------------------------------------------- #
    parser.add_argument("--max_seq_length", default=512, type=int, help="文本最长长度")
    parser.add_argument("--model_name", default="bert-base-chinese", type=str, help="预训练模型的名字")
    
    # --------------------------------------------- #
    # 4.训练参数;
    # --------------------------------------------- #
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=8, type=int, help="Batch size for evaluation.")
    parser.add_argument("--lr", default=5e-5, type=float, help="The initial learning rate for Optimizer.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--eva_step", default=100, type=int, help="Every X steps to evaluate.")
    
    # --------------------------------------------- #
    # 5.其他参数;
    # --------------------------------------------- #
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_valid", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization")
    parser.add_argument("--use_mutil_gpu", action="store_true", help="Whether to use more than 1 gpu.")
    parser.add_argument("--gpu_idx", default=0, help="GPU's num.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Whether to rewrite the cached dataset.")
    parser.add_argument(
        "--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        "--fp16_opt_level", type=str, default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    )
    
    args = parser.parse_args()
    
    save_path = os.path.join(args.output_path, args.save_name)
    if os.path.exists(save_path) and os.listdir(save_path) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(save_path))
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    # --------------------------------------------- #
    # Setting Logger
    # --------------------------------------------- #
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    
    handler = logging.FileHandler(args.output_path + "/" + args.save_name + "/log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARN)

    if args.use_mutil_gpu:
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16
        )
        raise ValueError("[×] 该方法尚未实现！")
    else:
        device = "cuda:{}".format(args.gpu_idx) if torch.cuda.is_available() else "cpu"
    args.device = device
    
    set_seed(args)
    
    labels = eval(get_labels(args))
    num_labels = len(labels)
    
    # --------------------------------------------- #
    # 使用交叉熵忽略索引(为`-100`)作为填充标签 ID，以便以后只有真实的标签 ID 有助于损失
    # --------------------------------------------- #
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    
    args.pretrained_model_path = os.path.join(args.ptm_path, args.model_name)
    model_config = AutoConfig.from_pretrained(args.pretrained_model_path, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=True)
    
    ner_model = BertCRFForTokenClassification.from_pretrained(args.pretrained_model_path, config=model_config)
    ner_model = ner_model.to(args.device)
    
    # --------------------------------------------- #
    # Training
    # --------------------------------------------- #
    if args.do_train:
        train_dataset = load_examples(args, args.train_data_name, tokenizer, labels, pad_token_label_id, mode="train")
        train_loader = DataLoader(
            train_dataset, args.train_batch_size, shuffle=True, collate_fn=lambda x: myFn(x, pad_token_label_id)
        )
        valid_dataset = load_examples(args, args.valid_data_name, tokenizer, labels, pad_token_label_id, mode="valid")
        valid_loader = DataLoader(
            valid_dataset, args.valid_batch_size, shuffle=False, collate_fn=lambda x: myFn(x, pad_token_label_id)
        )
        train(args, train_loader, valid_loader, ner_model, tokenizer, labels, pad_token_label_id)

if __name__ == "__main__":
    main()