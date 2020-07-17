from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
import torch
from .tokenizer import pad_token, unk_token, spc_token
from tqdm import tqdm


class FormalDataLoader(object):
    def __init__(self, data_path, tokenizer, test_path=None, max_token_len=64):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        if test_path is not None:
            self.test_path = test_path

        return

    def get_train_valid_dataset(self, test_size=0.1):
        data = get_data(self.data_path)
        train, valid = train_test_split(data, test_size=test_size, shuffle=True)
        train = self._get_dataset(train, self.max_token_len)
        valid = self._get_dataset(valid, self.max_token_len)
        return train, valid

    def get_test_dataset(self):
        if self.test_path is not None:
            data = get_data(self.test_path)
        else:
            data = get_data(self.data_path)

        data = self._get_dataset(data, self.max_token_len)
        return data

    def _get_dataset(self, data, max_token_len=64):
        token_ids, lenghts, labels = self._get_feature(data, max_token_len)

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        lenghts = torch.tensor(lenghts, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return TensorDataset(token_ids, lenghts, labels)

    def _get_feature(self, data, max_token_len=64):

        token_ids = []
        lenghts = []
        labels = []

        for sentence in tqdm(data, desc='convert feature'):
            temp_ids, temp_labels, temp_length = self.tokenizer.parse(sentence, max_token_len)
            token_ids.append(temp_ids)
            lenghts.append(temp_length)
            labels.append(temp_labels)

        return token_ids, lenghts, labels

class MyDataLoader(object):
    def __init__(self, data_path, tokenizer, test_path=None, max_token_len=64):

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

        if test_path is not None:
            self.test_path = test_path

        return

    def get_train_valid_dataset(self, test_size=0.1):
        data = get_data(self.data_path)
        train, valid = train_test_split(data, test_size=test_size, shuffle=True)
        train = self._get_dataset(train, self.max_token_len)
        valid = self._get_dataset(valid, self.max_token_len)
        return train, valid

    def get_test_dataset(self):
        if self.test_path is not None:
            data = get_data(self.test_path)
        else:
            data = get_data(self.data_path)

        data = self._get_dataset(data, self.max_token_len)
        return data

    def _get_dataset(self, data, max_token_len=64):
        token_ids, lenghts, labels = self._get_feature(data, max_token_len)

        token_ids = torch.tensor(token_ids, dtype=torch.long)
        lenghts = torch.tensor(lenghts, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        return TensorDataset(token_ids, lenghts, labels)

    def _get_feature(self, data, max_token_len=64):

        token_ids = []
        lenghts = []
        labels = []

        for sentence in tqdm(data, desc='convert feature'):
            temp_ids, temp_labels = self.tokenizer.parse(sentence, max_token_len)
            temp_ids = temp_ids[:max_token_len]
            temp_labels = temp_labels[:max_token_len]
            temp_len = len(temp_ids)

            if len(temp_ids) < max_token_len:
                pad = [self.tokenizer.get_id(pad_token)] * (max_token_len - len(temp_ids))
                temp_ids += pad
                temp_labels += pad

            token_ids.append(temp_ids)
            lenghts.append(temp_len)
            labels.append(temp_labels)

        return token_ids, lenghts, labels


def get_data(path, bylines=True):
    with open(path, 'r') as f:
        if bylines:
            return [line.rstrip('\n') for line in f.readlines()]
        else:
            return f.read()


if __name__ == "__main__":
    print("main")
