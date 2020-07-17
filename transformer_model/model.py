import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

## It is reconsturcted by using resources below
## 1. https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py
## 2. https://pytorch.org/tutorials/beginner/transformer_tutorial.html

class Transformer_TAG(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.tag_size = config.tag_size

        ## Transformer
        self.transformer = Transformer(config)

        ## Output
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(config.hidden_dim, config.tag_size)
        self.softmax = nn.LogSoftmax(dim=-1)

        self.crit = nn.NLLLoss()


    def forward(self, input_ids, lengths, tags=None):

        hiddens = self.transformer(input_ids)
        hiddens = self.tanh(self.fc(hiddens))
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

        ps = torch.exp(feats)
        top_p, top_class = ps.topk(1, dim=2)
        label_hat = top_class.squeeze(2).tolist()

        return label_hat

    def get_prob(self, input_ids, lengths):
        hiddens = self.transformer(input_ids)
        hiddens = self.tanh(self.fc(hiddens))
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps


class Transformer_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pad_token_id = config.pad_token_id
        self.tag_size = config.tag_size

        ## Transformer
        self.transformer = Transformer(config)

        ## Output
        self.fc = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.tanh = nn.Tanh()
        self.hidden2tag = nn.Linear(config.hidden_dim, config.tag_size)
        self.crf = CRF(config.tag_size, batch_first=True)

    def forward(self, input_ids, lengths, tags=None):
        hiddens = self.transformer(input_ids)
        hiddens = self.tanh(self.fc(hiddens))
        feats = self.hidden2tag(hiddens)

        mask = input_ids.ne(self.pad_token_id)
        if tags is not None:
            ## Remove Pad
            pack_tags = pack(tags,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)
            tags, _ = unpack(pack_tags, batch_first=True)

            ## Remove Pad
            pack_mask = pack(mask,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)
            mask, _ = unpack(pack_mask, batch_first=True)

            ## Remove Pad
            pack_feats = pack(feats,
                             lengths=lengths.tolist(),
                             batch_first=True,
                             enforce_sorted=False)
            feats, _ = unpack(pack_feats, batch_first=True)

            return -self.crf(feats, tags, mask=mask, reduction='sum')

        return self.crf.decode(feats, mask=mask)


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pad_token_id = config.pad_token_id
        self.device = config.device

        ## Embedding
        self.embedding = Embedding(config)

        ## Encoder
        self.encoders = nn.ModuleList([Encoder(config) for _ in range(config.num_layers)])

    def forward(self, input_ids, position_ids=None, attention_mask=None):
        # |input_ids| = (batch_size, token_len)
        input_shape = input_ids.size()
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if attention_mask is None:
            attention_mask = input_ids.eq(self.pad_token_id)

        hiddens = self.embedding(input_ids, position_ids).permute(1, 0, 2)

        for _, encoder in enumerate(self.encoders):
            hiddens = encoder(hiddens, attention_mask)

        return hiddens.permute(1, 0, 2)


class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()

        vocab_size = config.vocab_size
        max_token_len = config.max_token_len
        hidden_dim = config.hidden_dim

        self.device = config.device

        ## Embedding
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = nn.Embedding(max_token_len, hidden_dim)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids, position_ids):
        inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_dim = config.hidden_dim
        intermediate_dim = config.intermediate_dim
        num_heads = config.num_heads

        self.atten = nn.MultiheadAttention(hidden_dim, num_heads, dropout=config.dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-12)

        self.feed_foward = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, hidden_dim)
        )

        self.dropout = nn.Dropout(config.dropout_prob)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-12)

    def forward(self, hiddens, attention_mask):

        hiddens, _ = self.atten(hiddens, hiddens, hiddens, key_padding_mask=attention_mask, attn_mask=None)
        hiddens = self.norm1(hiddens)
        hiddens = self.feed_foward(hiddens)
        hiddens = self.dropout(hiddens)
        hiddens = self.norm2(hiddens)

        return hiddens

