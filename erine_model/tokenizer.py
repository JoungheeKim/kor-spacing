from collections import Counter
from transformers import BertTokenizer
from soynlp.tokenizer import MaxScoreTokenizer
from soynlp.noun import LRNounExtractor_v2
from gensim.models import Word2Vec
import torch
import numpy as np
import pickle
import os
pad_token = '<PAD>'
unk_token = '<UNK>'
spc_token = ' '


class MyIterator:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        if type(self.path) == str:
            with open(self.path, encoding='utf-8') as file:
                for sentence in file:
                    sentence = sentence.replace('\n', '').strip()
                    yield sentence
        else:
            for sentence in self.path:
                sentence = sentence.replace('\n', '').strip()
                yield sentence


class Word2VecCorpus:
    def __init__(self, path, tokenizer):
        self.path = path
        self.tokenizer = tokenizer

    def __iter__(self):
        if type(self.path) == str:
            with open(self.path, encoding='utf-8') as file:
                for sentence in file:
                    sentence = sentence.replace('\n', '').strip()
                    tokens = self.tokenizer.tokenize(sentence)
                    tokens = [token for token in tokens if token in self.tokenizer._scores]
                    yield tokens
        else:
            for sentence in self.path:
                sentence = sentence.replace('\n', '').strip()
                tokens = self.tokenizer.tokenize(sentence)
                tokens = [token for token in tokens if token in self.tokenizer._scores]
                yield tokens

def build_vocab(config, data=None):
    if data is not None:
        sents = MyIterator(data)
    else:
        sents = MyIterator(config.data_path)

    noun_extractor = LRNounExtractor_v2(verbose=False)
    nouns = noun_extractor.train_extract(sents)

    noun_dict = {}
    for noun, score in nouns.items():
        if score.frequency >= config.min_frequency and score.score >= config.min_score and len(noun) > config.min_length:
            noun_dict[noun] = score.score

    vocab_path = os.path.join(config.save_path,'vocab.pkl')
    config.vocab_path = vocab_path
    #save_pickle(vocab_path, noun_dict)

    tokenizer = MaxScoreTokenizer(noun_dict)

    if data is not None:
        word2vec_corpus = Word2VecCorpus(data, tokenizer)
    else:
        word2vec_corpus = Word2VecCorpus(config.data_path, tokenizer)

    word2vec_model = Word2Vec(
        word2vec_corpus,
        size=config.word_hidden_size,
        alpha=0.025,
        window=5,
        min_count=config.min_frequency,
        sg=0,
        negative=5)

    word2vec_path = os.path.join(config.save_path, 'word2vec{}.model'.format(config.word_hidden_size))
    config.word2vec_path = word2vec_path
    #word2vec_model.save(word2vec_path)

    return noun_dict, word2vec_model

def open_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def save_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    return True


class ErineTokenizer(object):
    def __init__(self, config):
        self.basic_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        if not ( os.path.isfile(config.vocab_path) and  os.path.isfile(config.word2vec_path) ):
            assert config.data_path and os.path.isfile(config.data_path), '[{}] 위치에 학습할 파일이 없습니다.'.format(config.data_path)
            noun_dict, word2vec_model = build_vocab(config)
        else:
            noun_dict = open_pickle(config.vocab_path)
            word2vec_model = Word2Vec.load(config.word2vec_path)

        self.config = config
        self.word_tokenizer = MaxScoreTokenizer(noun_dict)

        self.index2word = [unk_token] + word2vec_model.wv.index2word
        word2index = {}
        for index, word in enumerate(self.index2word):
            word2index[word] = index

        self.word2index = word2index
        self.pad_word_id = 0

        self.word_vec_dim = word2vec_model.vector_size
        self.word_vocab_size = len(word2index)

        unknown_emb = np.zeros((1, self.word_vec_dim), dtype=float)
        embedding = word2vec_model.wv.vectors
        self.embedding = torch.from_numpy(np.concatenate([unknown_emb, embedding], axis=0).astype(np.float))

        self.idx2tags = ['I', 'B']
        self.tags2idx = {name: idx for idx, name in enumerate(self.idx2tags)}
        self.tag_size = len(self.idx2tags)
        self.vocab_size = self.basic_tokenizer.vocab_size

        self.pad_token_id = self.basic_tokenizer.pad_token_id
        self.unk_token_id = self.basic_tokenizer.unk_token_id

    def reset_tokenizer(self, data):
        noun_dict, word2vec_model = build_vocab(self.config, data)
        self.word_tokenizer = MaxScoreTokenizer(noun_dict)

        self.index2word = [unk_token] + word2vec_model.wv.index2word
        word2index = {}
        for index, word in enumerate(self.index2word):
            word2index[word] = index

        self.word2index = word2index
        self.pad_word_id = 0

        self.word_vec_dim = word2vec_model.vector_size
        self.word_vocab_size = len(word2index)

        unknown_emb = np.zeros((1, self.word_vec_dim), dtype=float)
        embedding = word2vec_model.wv.vectors
        self.embedding = torch.from_numpy(np.concatenate([unknown_emb, embedding], axis=0).astype(np.float))



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

    def word_encode(self, sentence, max_length=None):
        sentence = sentence.replace(spc_token, '')
        word_tokens = self.word_tokenizer.tokenize(sentence)

        word_token_ids = []
        for idx, word in enumerate(word_tokens):

            temp_ids = []
            if word in self.word2index:
                temp_ids.append(self.word2index[word])

            padding = [self.pad_word_id] * (len(word) - len(temp_ids))
            temp_ids = temp_ids + padding

            word_token_ids.extend(temp_ids)

        if max_length is not None:
            word_token_ids = word_token_ids[:max_length]
            padding = [self.pad_word_id] * (max_length - len(word_token_ids))
            word_token_ids = word_token_ids + padding

        return word_token_ids


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
            labels = labels[:max_length]

        labels = labels
        length = len(labels)

        if max_length is not None and len(labels) < max_length:
            pad = [0] * (max_length - len(labels))
            labels = labels + pad
        return labels, length

    def parse(self, sentence: str, max_length=None):
        sentence = sentence.strip()
        tokens = self.tokenize(sentence)
        token_ids = self.encode(tokens, max_length)
        word_token_ids = self.word_encode(sentence, max_length)
        lables, length = self.get_labels(tokens, max_length)
        return token_ids, word_token_ids, lables, length

    def get_id(self, token: str):
        if token in self.basic_tokenizer.get_vocab():
            return self.basic_tokenizer.get_vocab()[token]
        else:
            return self.basic_tokenizer.unk_token_id
