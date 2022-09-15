import os
import torch
import logging
import jsonlines

from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


class InputExample(object):

    def __init__(self, guid, words, labels, candidates):
        self.guid = guid
        self.words = words
        self.labels = labels
        self.candidates = candidates


class InputFeatures(object):

    def __init__(self, input_ids, input_mask, label_ids, candi_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_ids = label_ids
        self.candi_mask = candi_mask


def read_examples_from_file(data_path, mode):
    """ [docs]
    return `examples`:
        :words: 就是分开后的 `item["content"]["tokens"]`
        如，[
            '2011', '年', '7', '月', '至', '2012', '年', '7', '月间', '，', '被告人',
            '王师才', '为', '非法', "获利", '，', '先后', "诱骗", '刘芬', '、',
            '杨武艳', '，', '或者', '通过', '被告人', '王超', "诱骗", '全忠萍', '至', '上海市', '奉贤区'
        ]
        :labels: 对应每个 token 的 label
        如，[
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'
            'O', 'O', 'O', "B-获利", 'O', 'O', "B-欺骗", 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', "B-欺骗", 'O', 'O', 'O', 'O'
        ]
        :candidates: 在event mention中被标注出来的token为1
        如，[
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0
        ]
    """
    input_data = jsonlines.open(data_path)  # 这句与原Code不一
    examples = list()
    
    for doc in input_data:
        words = [c['tokens'] for c in doc['content']]
        labels = [['O']*len(c['tokens']) for c in doc['content']]
        candidates = [[0]*len(c['tokens']) for c in doc['content']]
        
        if mode != "test":
            for event in doc['events']:
                for mention in event['mention']:
                    labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                    candidates[mention['sent_id']][mention['offset'][0]] = 1
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "I-" + event['type']
                        candidates[mention['sent_id']][i] = 1
                        
            # -------------------------------------------- #
            # 如果极性为 negative 的 Trigger 则不被预测出来
            # -------------------------------------------- #
            for mention in doc['negative_triggers']:
                labels[mention['sent_id']][mention['offset'][0]] = "O"
                candidates[mention['sent_id']][mention['offset'][0]] = 1
                for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                    labels[mention['sent_id']][i] = "O"
                    candidates[mention['sent_id']][i] = 1

        for i in range(0, len(words)):
            examples.append(
                InputExample(
                    guid   = "%s-%s-%d" % (mode, doc['id'], i),
                    words  = words[i],
                    labels = labels[i],
                    candidates = candidates[i]
                )
            )
            
    return examples


def convert_examples_to_features(examples, label_list, tokenizer, pad_token_label_id, max_seq_length):
    """ [docs]
    return `features`:
        :label_ids:  经过 tokenizer 分词后每个 token 的第一个会被打上标签，后续以及特殊符号打上-100;
        :candi_mask: 被标注为 candidates 的文本进行 mask，只有这个token的第一个字会mask为1，其他为0;
    """
    label2id = { label: i for i, label in enumerate(label_list) }

    features = list()
    for example in tqdm(examples):
        tokens, label_ids, candidates_mask = list(), list(), list()
        for word, label, candidate in zip(example.words, example.labels, example.candidates):
            word_tokens = tokenizer.tokenize(word)
            
            if len(word_tokens) == 0:
                word_tokens = ["[UNK]"]
            tokens.extend(word_tokens)
            
            if candidate == 1:
                label_ids.extend([label2id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
            else:
                label_ids.extend([pad_token_label_id] * len(word_tokens))

            candidates_mask.extend([candidate] + [0] * (len(word_tokens) - 1))

        if len(label_ids) > max_seq_length - 2 and len(tokens) > max_seq_length - 2:
            tokens          = tokens[:(max_seq_length - 2)]
            label_ids       = label_ids[:(max_seq_length - 2)]
            candidates_mask = candidates_mask[:(max_seq_length - 2)]
        
        assert len(label_ids) == len(tokens) == len(candidates_mask), "Token、Label以及candidates_mask长度不匹配！"

        input_ids = torch.tensor(tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"]), dtype=torch.long)
        input_mask = torch.ones_like(input_ids, dtype=torch.long)
        label_ids = torch.tensor([pad_token_label_id] + label_ids + [pad_token_label_id], dtype=torch.long)
        candidates_mask = torch.tensor([0] + candidates_mask + [0], dtype=torch.long)

        assert input_ids.shape == input_mask.shape == label_ids.shape == candidates_mask.shape

        features.append(
            InputFeatures(
                input_ids   = input_ids,
                input_mask  = input_mask,
                label_ids   = label_ids,
                candi_mask  = candidates_mask
            )
        )
        
    return features


# -------------------------------------------- #
# crf_array 就是每个原生token对应一个标签
# crf_pad   就是末尾pad的部分的mask为false
# -------------------------------------------- #
def to_crf_pad(org_array, org_mask, pad_label_id):
    """ [docs]
    The viterbi decoder function in CRF makes use of multiplicative property of 0,
    then pads wrong numbers out. Need a*0 = 0 for CRF to work.
    """
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask) if bb.any()]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    
    crf_pad = (crf_array != pad_label_id)
    crf_array[~crf_pad] = 0

    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_mask, pad_token_label_id):
    out_array = torch.ones(
        org_mask.shape, dtype=torch.int64, device=returned_mask.device
    ) * pad_token_label_id
    out_array[org_mask] = returned_array[returned_mask]
    
    return out_array


def check_file_path(args):
    save_path = args.output_path + "/" + args.save_name + "-Stage2[]-Dev[]"
    if os.path.exists(save_path) and os.listdir(save_path) and args.do_train:
        yesOrNo = input("输出文件夹({})已存在且非空，是否想要删除？[yes/no]".format(save_path))
        if yesOrNo == "yes":
            os.system("rm -r {}".format(save_path))
        elif yesOrNo == "no":
            raise ValueError("输出文件夹({})已存在，请手动操作！".format(save_path))
        else:
            raise ValueError("输出文件夹({})已存在且非空！".format(save_path))
    if not os.path.exists(save_path):
        os.mkdir(save_path)


def setting_logger(logger, args):
    logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
            
    handler = logging.FileHandler(args.output_path + "/" + args.save_name + "-Stage2[]-Dev[]" + "/log.txt")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(handler)
    
    logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)
    logging.getLogger("transformers.tokenization_utils").setLevel(logging.WARN)


def print_cfg(logger, len_train_loader, total_step, args):
    logger.info("================= Running Training =================")
    logger.info("- - - - - - - - - Basic Arguments  - - - - - - - - -")
    logger.info("    Seed                 = %d", args.seed)
    logger.info("    Num Train Loader     = %d", len_train_loader)
    logger.info("    Total Step           = %d", total_step)
    logger.info("    Data Fold n          = %s", args.data_fold)
    logger.info("    Model Save Path      = %s", args.output_path + "/"+ args.save_name + "-Stage2[]-Dev[]" )
    logger.info("    PRF Save Path        = %s", f"{args.output_path}/{args.save_name}-Stage2[]-Dev[]/prf_report_dict.json")
    logger.info("    CLS Save Path        = %s", f"{args.output_path}/{args.save_name}-Stage2[]-Dev[]/prf_report_text.txt")
    logger.info("- - - - - - - - - Model Arguments - - - - - - - - --")
    logger.info("    Model Name           = %s", args.model_name)
    logger.info("    Using CRF            = %s", str(args.use_crf))
    logger.info("    Using Last n Layers  = %d", args.last_n)
    if args.last_n > 1:
        logger.info("    Pooling Type(n > 1)  = %s", args.pooling_type)
    logger.info("    Using Focal Loss     = %s", str(args.use_focal))
    if args.use_focal:
        logger.info("    Focal Loss Weight    = %s", args.focal_weight)
        logger.info("    Focal Loss Gamma     = %s", args.focal_gamma)
        logger.info("    Focal Loss Alpha     = %s", args.focal_alpha)
        logger.info("    Focal Begin From     = %s", args.focal_begin_epoch)
    logger.info("    Multi-Sample Dropout = %s", str(args.MSD))
    logger.info("- - - - - - - - - Training Arguments - - - - - - - -")
    logger.info("    Num Epochs           = %d", args.num_train_epochs)
    logger.info("    Eva Begin Epoch      = %d", args.eval_begin_epoch)
    logger.info("    Max Seq Length       = %d", args.max_seq_length)
    logger.info("    Train Batch Size     = %d", args.train_batch_size)
    logger.info("    Valid Batch Size     = %d", args.valid_batch_size)
    logger.info("    Evaluation Step      = %d", args.eva_step)
    logger.info("    Whether Use AWP      = %d", args.use_awp)
    if args.use_awp:
        logger.info("    AWP Adversarial lr   = %s", args.awp_adv_lr)
        logger.info("    AWP Adversarial eps  = %s", args.awp_adv_eps)
        logger.info("    AWP Begin Epoch      = %s", args.awp_start_epoch)
        logger.info("    AWP Attack n once    = %s", args.awp_adv_step)
    logger.info("- - - - - - - - - Optimal Arguments - - - - - - - --")
    logger.info("    Warm-up Type         = %s", args.warmup_type)
    logger.info("    Warmup Prop          = %f", args.warmup_proportion)
    logger.info("    Adam Epsilon         = %s", str(args.adam_epsilon))
    logger.info("    Learning Rate        = %f", args.lr)
    logger.info("    Weight Decay         = %f", args.weight_decay)
    logger.info("- - - - - - - - - SpeedUp Arguments - - - - - - - --")
    logger.info("    Using Fp16           = %s", str(args.fp16))
    logger.info("    Fp16 Opt Level       = %s", args.fp16_opt_level)
    logger.info("- - - - - - - - - Other Arguments - - - - - - - - --")
    logger.info("    Do save model        = %s", str(args.do_save))
    logger.info("    Do save data         = %s", str(args.do_save_data))
    if args.early_stop_epoch == -1:
        logger.info("    Early Stop Epoch     = %s\n", str(False))
    else:
        logger.info("    Early Stop Epoch     = %d\n", args.early_stop_epoch)


class AWP:
    
    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.2, start_epoch=0, adv_step=1):
        
        self.model       = model
        self.optimizer   = optimizer
        self.adv_param   = adv_param
        self.adv_lr      = adv_lr
        self.adv_eps     = adv_eps
        self.start_epoch = start_epoch
        self.adv_step    = adv_step
        self.backup      = dict()
        self.backup_eps  = dict()

    def attack_backward(self, **batch):
        if (self.adv_lr == 0) or (batch["epoch"] < self.start_epoch):
            return None

        self._save()
        for _ in range(self.adv_step):
            self._attack_step()
            adv_loss = self.model(**batch).mean()
            self.optimizer.zero_grad()
            adv_loss.backward()
            
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )
                param.data.clamp_(*self.backup_eps[name])

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup     = dict()
        self.backup_eps = dict()