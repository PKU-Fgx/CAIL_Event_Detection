import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import json
import torch
import argparse
import jsonlines
import numpy as np

from tqdm import tqdm
from model import ModelForTokenClassification

from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer
)


parser = argparse.ArgumentParser()

parser.add_argument("--last_n",       default=1,      type=int,   help="用最后几层进行池化")
parser.add_argument("--pooling_type", default="mean", type=str,   help="池化方式（n=1时失效）")
parser.add_argument("--MSD",          default=False,  type=bool,  help="是否使用Multi-Sample Dropout")
parser.add_argument("--use_crf",      default=True,   type=bool,  help="是否使用CRF")
parser.add_argument("--gpu_idx",      default=0,      type=int,   help="GPU ID")
parser.add_argument("--model_saved",  default="None", type=str,   help="模型保存名称")
parser.add_argument("--use_focal",    default=False,  type=bool,  help="是否使用focal loss(infer 阶段无用)")
parser.add_argument("--do_save_data", action ="store_true", help="是否保存 test data")

args = parser.parse_args()

# -============ BEGIN =============- #
#    每次运行时可能需要更改的参数
# -============ BEGIN =============- #
device = "cuda:{}".format(args.gpu_idx)
model_type = args.model_saved
# -============= END ==============- #

max_seq_length = 512
test_batch_size = 16

test_data_name = "test_stage2.jsonl"
data_root_path = "/tf/FangGexiang/1.CAILED/Data"
listLabel_path = "/tf/FangGexiang/1.CAILED/Data/label.txt"
bio_label_path = "/tf/FangGexiang/1.CAILED/Data/bio_label.txt"

pretrained_model_path = "/tf/FangGexiang/1.CAILED/ModelSaved/" + model_type
output_test_predictions_file = pretrained_model_path + f"/results-Stage2.jsonl"


class myDataset(Dataset):
    
    def __init__(self, features):
        self.features = features
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def __len__(self):
        return len(self.features)

    
class InputExample(object):

    def __init__(self, guid, words, labels):
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids


def read_examples_from_file(data_path):
    input_data = jsonlines.open(data_path)
    examples = list()
    
    for doc in input_data:
        words = [c['tokens'] for c in doc['content']]
        labels = [['O']*len(c['tokens']) for c in doc['content']]

        for i in range(0, len(words)):
            examples.append(
                InputExample(
                    guid   = "%s-%d" % (doc['id'], i),
                    words  = words[i],
                    labels = labels[i]
                )
            )
            
    return examples

def convert_examples_to_features(examples, label_list, tokenizer, pad_token_label_id, max_seq_length):
    label2id = { label: i for i, label in enumerate(label_list) }

    features = list()
    for example in tqdm(examples):
        tokens, label_ids = list(), list()
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) == 0:
                tokens.extend(["[UNK]"])
                label_ids.extend([label2id[label]])
            else:
                tokens.extend(word_tokens)
                label_ids.extend([label2id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if len(label_ids) > max_seq_length - 2 and len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
            label_ids = label_ids[:(max_seq_length - 2)]
        
        assert len(label_ids) == len(tokens), "Token长度和Label长度不匹配！"

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        label_ids = torch.tensor([pad_token_label_id] + label_ids + [pad_token_label_id], dtype=torch.long)
        
        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens), dtype=torch.long)
        input_mask = torch.ones_like(input_ids, dtype=torch.long)

        assert input_ids.shape == input_mask.shape == label_ids.shape

        features.append(
            InputFeatures(
                input_ids   = input_ids,
                input_mask  = input_mask,
                label_ids   = label_ids
            )
        )
        
    return features


def load_examples(data_name, tokenizer, labels, pad_token_label_id, max_seq_length):
    cached_features_file = "/tf/FangGexiang/1.CAILED/Data/Cache/cached_{}_infer_{}".format(model_type, max_seq_length)
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s" % cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s" % os.path.join(data_root_path, data_name))
        examples = read_examples_from_file(os.path.join(data_root_path, data_name))
        features = convert_examples_to_features(
            examples, labels, tokenizer, pad_token_label_id=pad_token_label_id, max_seq_length=max_seq_length
        )

        # print("Saving features into cached file %s" % cached_features_file)
        # torch.save(features, cached_features_file)

    return myDataset(features)


def evaluate(device, loader, model, tokenizer, label_list, pad_token_label_id):
    """ Evaluation """
    model.eval()

    preds, out_label_ids = list(), list()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating..."):
            batch = { k: v.to(device) for k, v in batch.items()}
            batch.update({ "pad_token_label_id": pad_token_label_id })
            labels = batch.pop("labels")
        
            best_path = model(**batch)

            preds.extend(best_path.tolist())
            out_label_ids.extend(labels.tolist())
    
    label_map = { i: label for i, label in enumerate(label_list) }
    
    preds_list = [[] for _ in range(len(out_label_ids))]
    
    for i in range(len(out_label_ids)):
        for j in range(len(out_label_ids[i])):
            if out_label_ids[i][j] != pad_token_label_id:
                preds_list[i].append(label_map[preds[i][j]])

    return preds_list


def myFn(batch, pad_token_label_id):
    input_ids = pad_sequence([feature.input_ids for feature in batch], batch_first=True)
    attention_mask = pad_sequence([feature.input_mask for feature in batch], batch_first=True)
    labels = pad_sequence([feature.label_ids for feature in batch], batch_first=True, padding_value=pad_token_label_id)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


labels = eval(open(bio_label_path, "r").readline())

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, use_fast=True)
model_config = AutoConfig.from_pretrained(pretrained_model_path, num_labels=len(labels))

ner_model = ModelForTokenClassification.from_pretrained(pretrained_model_path, config=model_config, args=args).to(device)

test_dataset = load_examples(test_data_name, tokenizer, labels, pad_token_label_id=-100, max_seq_length=max_seq_length)
test_loader = DataLoader(
    test_dataset, test_batch_size, shuffle=False, collate_fn=lambda x: myFn(x, -100)
)
predictions = evaluate(device, test_loader, ner_model, tokenizer, labels, -100)

pure_event2id = eval(open(listLabel_path, "r").readline())
with open(output_test_predictions_file, "w") as writer:
    Cnt = 0
    levenTypes = list(pure_event2id.keys())
    with open(os.path.join(data_root_path, test_data_name), "r") as fin:
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="Writing..."):
            doc = json.loads(line)
            res = {}
            res['id'] = doc['id']
            res['predictions'] = []
            
            for mention in doc['candidates']:
                if mention['offset'][1] > len(predictions[Cnt + mention['sent_id']]):
                    res['predictions'].append({"id": mention['id'], "type_id": 0})
                    continue
                    
                is_NA = False if predictions[Cnt + mention['sent_id']][mention['offset'][0]].startswith("B") else True
                
                if not is_NA:
                    Type = predictions[Cnt + mention['sent_id']][mention['offset'][0]][2:]
                    
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        if predictions[Cnt + mention['sent_id']][i][2:] != Type:
                            is_NA = True
                            break
                            
                    if not is_NA:
                        res['predictions'].append({"id": mention['id'], "type_id": levenTypes.index(Type)})
                        
                if is_NA:
                    res['predictions'].append({"id": mention['id'], "type_id": 0})
                    
            writer.write(json.dumps(res) + "\n")
            Cnt += len(doc['content'])