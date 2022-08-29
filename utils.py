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


def convert_examples_to_features(examples, label_list, tokenizer, pad_token_label_id, max_seq_length):
    """ [docs]
    return `features`:
        :label_ids: 经过 tokenizer 分词后每个 token 的第一个会被打上标签，后续以及特殊符号打上-100;
    """
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
