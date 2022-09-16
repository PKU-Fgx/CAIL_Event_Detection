import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import warnings
warnings.filterwarnings("ignore")

import time
import jsonlines

from tqdm import tqdm
from accelerate import Accelerator
from tensorboardX import SummaryWriter
from dataclasses import dataclass, field
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup
)


@dataclass
class Augments:
    train_path : str                  = field(default="/tf/FangGexiang/1.CAILED/Data/MutilFold/fold_2/train.jsonl",     metadata={ "help": "训练文件路径" })
    valid_path : str                  = field(default="/tf/FangGexiang/1.CAILED/Data/MutilFold/fold_2/valid.jsonl",     metadata={ "help": "验证集文件路径" })
    infer_path : str                  = field(default="/tf/FangGexiang/1.CAILED/Data/test.jsonl",                       metadata={ "help": "测试集文件路径" })
    test1_path : str                  = field(default="/tf/FangGexiang/1.CAILED/Data/test_stage1.jsonl",                metadata={ "help": "测试阶段1文件路径" })
    test2_path : str                  = field(default="/tf/FangGexiang/1.CAILED/Data/test_stage2.jsonl",                metadata={ "help": "测试阶段2文件路径" })
    pretrained_model_path : str       = field(default="/tf/FangGexiang/3.SememeV2/pretrained_model/bert-base-chinese",  metadata={ "help": "预训练模型的位置" })
    domain_pretrained_save_path : str = field(default="/tf/FangGexiang/1.CAILED/domainPretrained/bert-base-cn-law",     metadata={ "help": "预训练的权重保存位置" })

    device : str                      = field(default="cuda:0",    metadata={ "help": "GPU idx" })
    batch_size : int                  = field(default=10,          metadata={ "help": "训练批次大小" })
    max_length : int                  = field(default=512,         metadata={ "help": "每一句话输入最长长度" })
    adam_epsilon : float              = field(default=1e-8,        metadata={ "help": "Epsilon for Adam optimizer." })
    warmup_proportion : float         = field(default=0.1,         metadata={ "help": "When to begin to warmup."})
    lr : float                        = field(default=5e-5,        metadata={ "help": "学习率" })
    epoches : int                     = field(default=20,          metadata={ "help": "迭代次数" })


def readFromReader(reader):
    """ 从 `json.reader` 类型中读出其中文本，返回一个 list """
    
    sentences_list = list()
    for doc in reader:
        for sentence in doc["content"]:
            sentences_list.append(sentence["sentence"])
    
    return sentences_list


def readFromFiles(file_path):
    """ 从文件中读取文本内容，输入`文件地址`，输出这个文件所有文本的列表 """

    if isinstance(file_path, str):
        contents = readFromReader(jsonlines.open(file_path))
    elif isinstance(file_path, list):
        contents = sum(list(map(lambda p: readFromReader(jsonlines.open(p)), file_path)), list())
    else:
        raise ValueError("`file_path` 参数异常，请检查！")
    
    return contents


class myDataset(Dataset):
    
    def __init__(self, contents):
        self.contents = contents
    
    def __getitem__(self, index):
        return self.contents[index]

    def __len__(self):
        return len(self.contents)


def myFn(batch, config, tokenizer, data_collator):
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=config.max_length, return_tensors="pt")
    input_ids, labels = data_collator.torch_mask_tokens(inputs.input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": inputs.attention_mask,
        "token_type_ids": inputs.token_type_ids,
        "labels": labels
    }


def train(config, accelerator, mlm_model, dataloader, tokenizer, my_writer=None):

    step, loss_list, total_step = 0, list(), len(dataloader) * config.epoches
    optimizer = AdamW(mlm_model.parameters(), eps=config.adam_epsilon, lr=config.lr)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_step * config.warmup_proportion), total_step)

    mlm_model, optimizer, dataloader, scheduler = accelerator.prepare(
        mlm_model, optimizer, dataloader, scheduler
    )

    for epoch in range(config.epoches):
        mlm_model.train()
        train_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        for batch in train_bar:
            optimizer.zero_grad()

            loss = mlm_model(**batch).loss

            accelerator.backward(loss)
            
            optimizer.step()
            scheduler.step()

            loss_list.append(loss.item())
            train_bar.set_description("[Epoch {:2d} / {:2d}] Loss: {:7.5f}".format(epoch, config.epoches, sum(loss_list)/len(loss_list)))

            if accelerator.is_local_main_process and my_writer is not None:
                my_writer.add_scalar('CurLoss', loss.item(), step)
                my_writer.add_scalar('AvgLoss', sum(loss_list)/len(loss_list), step)
            step += 1

        # -------------- #
        #     Saving     #
        # -------------- #
        mlm_model = accelerator.unwrap_model(mlm_model)
        mlm_model.save_pretrained(
            config.domain_pretrained_save_path + "Epoch[{}]".format(epoch),
            is_main_process=accelerator.is_local_main_process,
            save_function=accelerator.save
        )
        tokenizer.save_pretrained(config.domain_pretrained_save_path + "Epoch[{}]".format(epoch))


def mlm():
    config = Augments()

    accelerator = Accelerator()

    accelerator.print("[1] 正在读取所有的文本文件 ...")
    contents = readFromFiles([config.train_path, config.valid_path, config.infer_path, config.test1_path, config.test2_path])

    accelerator.print("[2] 构建 Tokenizer 与 Model ...")
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_path)
    mlm_model = AutoModelForMaskedLM.from_pretrained(config.pretrained_model_path)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer)

    dataset = myDataset(contents)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=lambda x: myFn(x, config, tokenizer, data_collator))

    if accelerator.is_local_main_process:
        current_time = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
        my_writer = SummaryWriter(log_dir=current_time)
        train(config, accelerator, mlm_model, dataloader, tokenizer, my_writer)
    else:
        train(config, accelerator, mlm_model, dataloader, tokenizer)


if __name__ == "__main__":
    mlm()