import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence as pack

class BERT_TAG(nn.Module):

    def __init__(self, config):
        super(BERT_TAG, self).__init__()

        self.device = config.device
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tag_size = config.tag_size
        self.tanh = nn.Tanh()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, self.tag_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crit = nn.NLLLoss()

    def forward(self, input_ids, lengths, tags=None):

        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)

        if tags is not None:
            ## Remove Pad
            pack_tags = pack(tags,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)

            pack_feats = pack(feats,
                              lengths=lengths.tolist(),
                              batch_first=True,
                              enforce_sorted=False)

            loss = self.crit(pack_feats.data, pack_tags.data)
            return loss

        return feats

    def get_prob(self, input_ids, lengths):
        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.tanh(self.fc(hiddens))
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps




