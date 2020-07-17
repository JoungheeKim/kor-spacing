from collections import Counter
from transformers import BertTokenizer
pad_token = '<PAD>'
unk_token = '<UNK>'
spc_token = ' '


class BITokenizer(object):
    def __init__(self):
        self.basic_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.idx2tags = ['I', 'B']
        self.tags2idx = {name: idx for idx, name in enumerate(self.idx2tags)}
        self.tag_size = len(self.idx2tags)
        self.vocab_size = self.basic_tokenizer.vocab_size

        self.pad_token_id = self.basic_tokenizer.pad_token_id
        self.unk_token_id = self.basic_tokenizer.unk_token_id

    def tokenize(self, sentence: str):
        tokens = []
        temp = ''
        for i in range(len(sentence)):
            if sentence[i] == spc_token:
                temp += sentence[i]
                continue

            temp += sentence[i]
            tokens.append(temp)
            temp = ''

        return tokens

    def encode(self, tokens: list, max_length=None):

        temp_tokens = []
        for token in tokens:
            token = token.replace(spc_token, '')
            temp_tokens.append(token)
        sentence = spc_token.join(temp_tokens)

        if max_length is not None:
            token_ids = self.basic_tokenizer.encode(sentence, max_length=max_length, pad_to_max_length=True, add_special_tokens=False, truncation=True)
        else:
            token_ids = self.basic_tokenizer.encode(sentence, add_special_tokens=False)

        return token_ids

    def decode(self, token_ids: list(), lables: list()):
        sentences = []
        for word_tokens, word_labels in zip(token_ids, lables):
            sentence = ""
            for token_id, label in zip(word_tokens, word_labels):
                if self.idx2tags[label] == 'B':
                    sentence += " "
                if token_id == self.basic_tokenizer.cls_token_id or token_id == self.basic_tokenizer.sep_token_id:
                    continue
                sentence += self.basic_tokenizer.convert_ids_to_tokens(token_id)
            sentences += [sentence]
        return sentences

    def get_labels(self, tokens: list, max_length=None):
        labels = [1 if spc_token in token else 0 for token in tokens]
        if max_length is not None:
            #labels = labels[:max_length-2]
            labels = labels[:max_length]
        
        ## max token 지우기
        #labels = [0] + labels + [0]
        labels = labels
        length = len(labels)

        if max_length is not None and len(labels) < max_length:
            pad = [0] * (max_length - len(labels))
            labels = labels + pad
        return labels, length

    def parse(self, sentences: str, max_length=None):
        sentences = sentences.strip()
        tokens = self.tokenize(sentences)
        token_ids = self.encode(tokens, max_length)
        lables, length = self.get_labels(tokens, max_length)
        return token_ids, lables, length

    def get_id(self, token: str):
        if token in self.basic_tokenizer.get_vocab():
            return self.basic_tokenizer.get_vocab()[token]
        else:
            return self.basic_tokenizer.unk_token_id

## https://arxiv.org/pdf/1807.02974.pdf 참고
class BIESTokenizer(BITokenizer):
    def __init__(self, vocab: dict):
        super(BIESTokenizer, self).__init__()
        self.idx2tags = ['S', 'B', 'I', 'E']
        self.tags2idx = {name: idx for idx, name in enumerate(self.idx2tags)}
        self.tag_size = len(self.idx2tags)

    def get_labels(self, tokens: list):
        labels = []

        seq_flag = False
        for idx, token in enumerate(tokens):
            if spc_token in token:
                if seq_flag:
                    labels[-1] = self.tags2idx['S']
                labels.append(self.tags2idx['B'])
                seq_flag = True
            else:
                if not seq_flag and idx>0:
                    labels[-1] = self.tags2idx['I']
                labels.append(self.tags2idx['E'])
                seq_flag = False
        return labels

