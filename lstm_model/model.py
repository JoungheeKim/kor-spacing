import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torchcrf import CRF

class BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()

        hidden_dim = config.hidden_dim
        vocab_size = config.vocab_size
        tag_size = config.tag_size
        num_layers = config.num_layers

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_size = tag_size
        self.num_layers = num_layers
        self.device = config.device


        self.word_embeds = nn.Embedding(vocab_size, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim // 2,
                            num_layers=num_layers, bidirectional=True, batch_first=True)

        self.tanh = nn.Tanh()

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, tag_size)

        self.crf = CRF(tag_size, batch_first=True)

    def _get_lstm_features(self, token_ids, lengths):
        # |token_ids| = [batch_size, token_length]
        # |lengths| = [batch_size]

        embeds = self.word_embeds(token_ids)
        # |embeds| = [batch_size, token_length, hidden_dim]
        packed_embeds = pack(embeds,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)
        # |embeds| = [batch_size, token_length, hidden_dim]

        # Apply RNN and get hiddens layers of each words
        last_hiddens, _ = self.rnn(packed_embeds)

        # Unpack ouput of rnn model
        last_hiddens, _ = unpack(last_hiddens, batch_first=True)
        # |last_hiddens| = [batch_size, max(token_length), hidden_size]
        lstm_feats = self.hidden2tag(self.tanh(last_hiddens))

        return lstm_feats

    def forward(self, token_ids, lengths, tags=None):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        # |token_ids| = [batch_size, token_length]
        # |lengths| = [batch_size]
        # |tags| = |batch_size, token_length|

        #print("token_ids.size()", token_ids.size())
        #print("lengths.size()", lengths.size())
        #print("lengths", lengths)

        lstm_feats = self._get_lstm_features(token_ids, lengths)
        mask = self._generate_mask(lengths)

        # Find the best path, given the features.
        if tags is not None:
            ## Remove Pad
            pack_tags = pack(tags,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)
            tags, _ = unpack(pack_tags, batch_first=True)
            return -self.crf(lstm_feats, tags, mask=mask, reduction='mean')

        return torch.tensor(self.crf.decode(lstm_feats, mask=mask), dtype=torch.long)

    def get_prob(self, token_ids, lengths):
        lstm_feats = self._get_lstm_features(token_ids, lengths)
        mask = self._generate_mask(lengths)

    def _generate_mask(self, length):
        mask = []

        max_length = max(length)
        for l in length:
            if max_length - l > 0:
                # If the length is shorter than maximum length among samples,
                # set last few values to be 1s to remove attention weight.
                mask += [torch.cat(
                    [torch.ones((1, l), dtype=torch.uint8), torch.zeros((1, (max_length - l)), dtype=torch.uint8)],
                    dim=-1)]
            else:
                # If the length of the sample equals to maximum length among samples,
                # set every value in mask to be 0.
                mask += [torch.ones((1, l), dtype=torch.uint8)]

        mask = torch.cat(mask, dim=0)

        return mask.to(self.device)