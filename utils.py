import torch
import jsonlines

from tqdm import tqdm


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


def read_examples_from_file(data_path, mode):
    """
    return `examples`:
        :words: 就是分开后的 `item["content"]["tokens"]`
        如，['2011', '年', '7', '月', '至', '2012', '年', '7', '月间', ...
        :labels: 对应每个 token 的 label
        如，['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', ...
    """
    input_data = jsonlines.open(data_path)  # 这句与原Code不一
    examples = list()
    
    for doc in input_data:
        words = [c['tokens'] for c in doc['content']]
        labels = [['O']*len(c['tokens']) for c in doc['content']]
        
        if mode != "test":
            for event in doc['events']:
                for mention in event['mention']:
                    labels[mention['sent_id']][mention['offset'][0]] = "B-" + event['type']
                    for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                        labels[mention['sent_id']][i] = "I-" + event['type']
                        
            # -------------------------------------------- #
            # 如果极性为 negative 的 Trigger 则不被预测出来
            # -------------------------------------------- #
            for mention in doc['negative_triggers']:
                labels[mention['sent_id']][mention['offset'][0]] = "O"
                for i in range(mention['offset'][0] + 1, mention['offset'][1]):
                    labels[mention['sent_id']][i] = "O"
            
        for i in range(0, len(words)):
            examples.append(
                InputExample(
                    guid   = "%s-%s-%d" % (mode, doc['id'], i),
                    words  = words[i],
                    labels = labels[i]
                )
            )
            
    return examples


def convert_examples_to_features(examples, label_list, tokenizer, pad_token_label_id, max_seq_length, sep_token_extra = False):
    """
    return `features`:
        :label_ids: 经过 tokenizer 分词后每个 token 的第一个会被打上标签，特殊符号打上-100;
    """
    label2id = { label: i for i, label in enumerate(label_list) }

    features = list()
    for example in tqdm(examples):
        tokens, label_ids = list(), list()
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            if len(word_tokens) != 0:
                label_ids.extend([label2id[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if sep_token_extra:
            if len(label_ids) > max_seq_length - 3:
                label_ids = label_ids[:(max_seq_length - 3)]
            label_ids += [pad_token_label_id] * 2
        else:
            if len(label_ids) > max_seq_length - 2:
                label_ids = label_ids[:(max_seq_length - 2)]
            label_ids += [pad_token_label_id]
        label_ids = [pad_token_label_id] + label_ids
        
        encoded_input = tokenizer(example.words, truncation=True, max_length=max_seq_length, is_split_into_words=True)
        input_ids = encoded_input.input_ids
        input_mask = encoded_input.attention_mask

        assert len(input_ids) == len(input_mask) == len(label_ids)

        features.append(
            InputFeatures(
                input_ids   = torch.tensor(input_ids, dtype=torch.long),
                input_mask  = torch.tensor(input_mask, dtype=torch.long),
                label_ids   = torch.tensor(label_ids, dtype=torch.long)
            )
        )
        
    return features
