from crf import *
from torch.nn.utils.rnn import pad_sequence
from transformers import BertConfig, BertModel, BertPreTrainedModel


def to_crf_pad(org_array, org_mask, pad_label_id):
    """ The viterbi decoder function in CRF makes use of multiplicative property of 0, then pads wrong numbers out.
        Need a*0 = 0 for CRF to work.
    """
    crf_array = [aa[bb] for aa, bb in zip(org_array, org_mask)]
    crf_array = pad_sequence(crf_array, batch_first=True, padding_value=pad_label_id)
    
    crf_pad = (crf_array != pad_label_id)
    crf_array[~crf_pad] = 0

    return crf_array, crf_pad


def unpad_crf(returned_array, returned_mask, org_array, org_mask):
    out_array = org_array.clone().detach()
    out_array[org_mask] = returned_array[returned_mask]
    
    return out_array


class BertCRFForTokenClassification(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.init_weights()

    def _get_features(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )

        feats = self.classifier(self.dropout(outputs.last_hidden_state))
        
        return feats

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        pad_token_label_id=None
    ):        
        logits = self._get_features(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds
        )
        
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)
            
            # Only keep active parts of the loss
            loss_mask = ((attention_mask == 1) & pad_mask) if attention_mask is not None else ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            # ---------------------------------------------------- #
            #   loss: Removing mask stuff from the output path is 
            # done later in my_crf_ner but it should be kept away
            # when calculating loss.
            # ---------------------------------------------------- #
            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            
            return loss
        else:
            # ---------------------------------------------------- #
            #   best_path: 长度和经过 tokenizer 分词好的 label 同
            # ---------------------------------------------------- #
            mask = (attention_mask == 1) if attention_mask is not None else torch.ones(logits.shape).bool()

            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape, dtype=torch.int64, device=crf_logits.device) * pad_token_label_id
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)

            return best_path