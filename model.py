import torch

from utils import *
from torch import nn
from torchcrf import CRF
from transformers import (
    PreTrainedModel,
    BertModel,
    NezhaModel,
    AlbertModel,
    RobertaModel,
    LongformerModel,
)


class ModelPreTrainedModel(PreTrainedModel):

    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"positions_encoding"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ModelForTokenClassification(ModelPreTrainedModel):
    
    def __init__(self, config, args):
        super(ModelForTokenClassification, self).__init__(config)
        self.args = args
        self.num_labels = config.num_labels
        self.model_type = config._name_or_path.split("/")[-1].split("-")

        if "nezha" in self.model_type:
            self.nezha = NezhaModel(config)
        elif "bert" in self.model_type:
            self.bert = BertModel(config)
        elif "roberta" in self.model_type:
            self.bert = BertModel(config)
        elif "Lawformer" in self.model_type:
            self.longformer = LongformerModel(config)
        else:
            raise ValueError("--model_name 参数异常, 在模型初始化时出错, 请检查!")
        
        if self.args.MSD:
            self.dropout = nn.ModuleList([nn.Dropout(config.hidden_dropout_prob * i) for i in range(1, 6)])
        else:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if self.args.use_crf:
            self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def get_features(self, input_ids=None, attention_mask=None, token_type_ids=None):
        if "nezha" in self.model_type:
            outputs = self.nezha(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif "bert" in self.model_type:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif "roberta" in self.model_type:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif "Lawformer" in self.model_type:
            outputs = self.longformer(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            raise ValueError("--model_name 参数异常, 在模型Foward时出错, 请检查!")
            
        if self.args.last_n == 1:
            features = outputs.last_hidden_state
        else:
            if self.args.pooling_type == "mean":
                model_outs = torch.stack(outputs.hidden_states[-1 * self.args.last_n:], dim=-1)
                features = torch.mean(model_outs, dim=-1)
            elif self.args.pooling_type == "max":
                model_outs = torch.stack(outputs.hidden_states[-1 * self.args.last_n:], dim=-1)
                features = torch.max(model_outs, dim=-1).values
            else:
                raise ValueError("--pooling_type 参数异常, 未知的池化方式!")
        
        return features

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, pad_token_label_id=None):
        features = self.get_features(input_ids, attention_mask, token_type_ids)

        if self.args.MSD:
            logits = torch.sum(self.classifier(torch.stack(list(map(lambda dp: dp(features) , self.dropout)))), dim=0)
        else:
            logits = self.classifier(self.dropout(features))
        
        if labels is not None:
            if self.args.use_crf:
                # ------------------------- #
                #   有用的就 Mask 为 True
                # ------------------------- #
                pad_mask = (labels != pad_token_label_id)
                
                if attention_mask is not None:
                    loss_mask = ((attention_mask == 1) & pad_mask)
                else:
                    loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

                crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
                crf_logits, _        = to_crf_pad(logits, loss_mask, pad_token_label_id)

                loss = -1.0 * self.crf(crf_logits, crf_labels, crf_mask)

                return loss
            else:
                loss_fn = nn.CrossEntropyLoss()

                return loss_fn(logits.view((-1, logits.shape[-1])), labels.view(-1))
        else:
            if self.args.use_crf:
                # ---------------------------------------------------- #
                #   best_path: 长度和经过 tokenizer 分词好的 label 同
                # ---------------------------------------------------- #
                mask = (attention_mask == 1) if attention_mask is not None else torch.ones(logits.shape).bool()

                # ---------------------------------------------------- #
                #   crf_mask: 获得那些位置上的logits不是0的，即有用的
                # ---------------------------------------------------- #
                crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
                crf_mask = (crf_mask.sum(axis=2) == crf_mask.shape[2])
                
                best_path = pad_sequence(
                    [torch.tensor(path, device=crf_logits.device) for path in self.crf.decode(crf_logits, crf_mask)],
                    batch_first=True, padding_value=0
                )

                return unpad_crf(best_path, crf_mask, mask, pad_token_label_id)
            else:
                return torch.argmax(logits, dim=-1)