import torch

from torch import nn
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PreTrainedModel,
    LongformerModel, LongformerPreTrainedModel,
    BertModel, BertPreTrainedModel,
    NezhaModel, NezhaPreTrainedModel,
    AlbertModel, AlbertPreTrainedModel,
    RobertaModel, RobertaPreTrainedModel,
    DebertaV2Model, DebertaV2PreTrainedModel
)

# -------------------------------------------- #
# crf_array 就是每个原生token对应一个标签
# crf_pad   就是末尾pad的部分的mask为false
# -------------------------------------------- #
def to_crf_pad(org_array, org_mask, pad_label_id):
    """ [docs]
    The viterbi decoder function in CRF makes use of multiplicative property of 0,
    then pads wrong numbers out. Need a*0 = 0 for CRF to work.
    """
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
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


class ModelPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

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
            

# ---------------------- CRF 款模型 ---------------------- #
class ModelCRFForTokenClassification(ModelPreTrainedModel):
    
    def __init__(self, config):
        super(ModelCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.model_type = config._name_or_path.split("/")[-1].split("-")

        if "nezha" in self.model_type:
            self.nezha = NezhaModel(config)
        elif "bert" in self.model_type:
            self.bert = BertModel(config)
        else:
            raise ValueError("--model_name 参数异常，在模型初始化时出错，请检查！")
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(self.num_labels, batch_first=True)

        self.init_weights()

    def get_features(self, input_ids=None, attention_mask=None, token_type_ids=None):
        if "nezha" in self.model_type:
            outputs = self.nezha(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif "bert" in self.model_type:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            raise ValueError("--model_name 参数异常，在模型Foward时出错，请检查！")
            
        features = self.classifier(self.dropout(outputs.last_hidden_state))
        
        return features

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, pad_token_label_id=None):        
        logits = self.get_features(input_ids, attention_mask, token_type_ids)
        
        if labels is not None:
            # ------------------------- #
            #   有用的就 Mask 为 True
            # ------------------------- #
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)
            
            # Only keep active parts of the loss
            if attention_mask is not None:
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _        = to_crf_pad(logits, loss_mask, pad_token_label_id)

            # ---------------------------------------------------- #
            #   loss: Removing mask stuff from the output path is 
            # done later in my_crf_ner but it should be kept away
            # when calculating loss.
            # ---------------------------------------------------- #
            loss = -1.0 * self.crf(crf_logits, crf_labels, crf_mask)
            
            return loss
        else:
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


# ---------------------- 经典款模型 ---------------------- #
class ModelForTokenClassification(ModelPreTrainedModel):
    
    def __init__(self, config):
        super(ModelForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.model_type = config._name_or_path.split("/")[-1].split("-")

        if "nezha" in self.model_type:
            self.nezha = NezhaModel(config)
        elif "bert" in self.model_type:
            self.bert = BertModel(config)
        else:
            raise ValueError("--model_name 参数异常，在模型初始化时出错，请检查！")
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.loss_fct = nn.CrossEntropyLoss()

        self.init_weights()

    def get_features(self, input_ids=None, attention_mask=None, token_type_ids=None):
        if "nezha" in self.model_type:
            outputs = self.nezha(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif "bert" in self.model_type:
            outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        else:
            raise ValueError("--model_name 参数异常，在模型Foward时出错，请检查！")
            
        features = self.classifier(self.dropout(outputs.last_hidden_state))
        
        return features

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, pad_token_label_id=None):        
        logits = self.get_features(input_ids, attention_mask, token_type_ids)
        
        if labels is not None:
            loss = self.loss_fct(logits.view((-1, logits.shape[-1])), labels.view(-1))
            return loss
        
        else:
            best_path = torch.argmax(logits, dim=-1)
            return best_path