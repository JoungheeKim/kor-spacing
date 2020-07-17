from collections import Counter
pad_token = '<PAD>'
unk_token = '<UNK>'
spc_token = ' '



## VOCAB 만들기
def build_vocab(path):
    with open(path, 'r') as f:
        data = f.read()
    vocab = dict(Counter(data))

    ## PAD, UNK, SPC token 추가
    idx2word = [pad_token, unk_token] + [token for token, count in vocab.items()]
    if spc_token not in idx2word:
        idx2word += [spc_token]

    word2idx = {token: idx for idx, token in enumerate(idx2word)}

    return {
        'idx2word': idx2word,
        'word2idx': word2idx
    }


class BITokenizer(object):
    def __init__(self, vocab: dict):
        self.vocab = vocab
        assert 'idx2word' in vocab and 'word2idx' in vocab, 'please rebuild vocab and try again'
        self.idx2word = vocab['idx2word']
        self.word2idx = vocab['word2idx']
        self.vocab_size = len(self.idx2word)

        self.idx2tags = ['I', 'B']
        self.tags2idx = {name: idx for idx, name in enumerate(self.idx2tags)}
        self.tag_size = len(self.idx2tags)

        self.pad_token_id = self.word2idx[pad_token]
        self.unk_token_id = self.word2idx[unk_token]

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

    def encode(self, tokens: list):
        token_ids = []
        for token in tokens:
            token = token.replace(spc_token, '')
            if token not in self.word2idx:
                token = unk_token
            token_ids += [self.word2idx[token]]

        return token_ids

    def decode(self, token_ids: list(), lables: list()):
        sentences = []
        for word_tokens, word_labels in zip(token_ids, lables):
            sentence = ""
            for token_id, label in zip(word_tokens, word_labels):
                if self.idx2tags[label] == 'B':
                    sentence += " "
                sentence += self.idx2word[token_id]
            sentences += [sentence]
        return sentences

    def get_labels(self, tokens: list):
        return [1 if spc_token in token else 0 for token in tokens]

    def parse(self, sentences: str):
        sentences = sentences.strip()
        tokens = self.tokenize(sentences)
        token_ids = self.encode(tokens)
        lables = self.get_labels(tokens)
        return token_ids, lables

    def get_id(self, token: str):
        if token in self.word2idx:
            return self.word2idx[token]
        else:
            return self.word2idx[unk_token]

## https://arxiv.org/pdf/1807.02974.pdf 참고
class BIESTokenizer(BITokenizer):
    def __init__(self, vocab: dict):
        super(BIESTokenizer, self).__init__(vocab)
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

