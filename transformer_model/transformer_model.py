import torch
import torch.nn as nn

from typing import Tuple, Dict

# This all implementations referred from huggingface
class Transformer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_embeddings, num_max_positions=512, num_heads=8, num_layers=4,
                 dropout=.1, causal=False):
        super().__init__()
        self.causal = causal
        self.tokens_embeddings = nn.Embedding(num_embeddings, embed_dim)
        self.position_embeddings = nn.Embedding(num_max_positions, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()

        for _ in range(num_layers):
            self.attentions.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.feed_forwards.append(nn.Sequential(nn.Linear(embed_dim, hidden_dim),
                                                    nn.ReLU(),
                                                    nn.Linear(hidden_dim, embed_dim)))
            self.layer_norms_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.tokens_embeddings(x)
        h = h + self.position_embeddings(positions).expand_as(h)
        h = self.dropout(h)

        attn_mask = None
        if self.causal:
            attn_mask = torch.full((len(x), len(x)), -float('Inf'), device=h.device, dtype=h.dtype)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h, h, h, attn_mask=attn_mask, need_weights=False, key_padding_mask=padding_mask)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h

        return h


class TransformerTagger(nn.Module):
    def __init__(self, vocab_size: int, tag_size: int, embedding_dim: int, hidden_dim: int, head_size: int,
                 layer_size: int, pad_idx=0) -> None:
        super(TransformerTagger, self).__init__()
        self._pad_idx = pad_idx
        self._tag_size = tag_size

        self.transformer = Transformer(embedding_dim, hidden_dim * head_size, vocab_size, num_heads=head_size,
                                       num_layers=layer_size, causal=True)
        self._fc1 = nn.Linear(embedding_dim, embedding_dim)
        self._activate = nn.Tanh()
        self._fc2 = nn.Linear(embedding_dim, tag_size)

        self._ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx)

        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        linear_hidden = self._activate(self._fc1(hiddens))
        emissions = self._fc2(linear_hidden)

        emissions = nn.functional.softmax(emissions, dim=-1)
        path = torch.argmax(emissions, dim=-1)
        max_prob = torch.max(emissions, dim=-1)[0]
        scores = torch.mean(max_prob)

        return scores, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx)
        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        linear_hidden = self._activate(self._fc1(hiddens))
        emissions = self._fc2(linear_hidden)

        nll = self._ce_loss(emissions.view(-1, self._tag_size), y.view(-1))
        # nll = torch.mean(nll * masking.view(-1).float())
        nll = torch.mean(nll)

        return nll


class TransformerCRF(nn.Module):
    def __init__(self, vocab_size: int, tag_to_idx: Dict, embedding_dim: int, hidden_dim: int, head_size: int,
                 layer_size: int, pad_idx=0) -> None:
        super(TransformerCRF, self).__init__()
        self._pad_idx = pad_idx
        self._tag_to_idx = tag_to_idx

        self.transformer = Transformer(embedding_dim, hidden_dim, vocab_size, num_heads=head_size,
                                       num_layers=layer_size)
        self._fc = nn.Linear(embedding_dim, len(tag_to_idx))

        self._crf = CRF(len(self._tag_to_idx), bos_tag_id=self._tag_to_idx[START_TAG],
                        eos_tag_id=self._tag_to_idx[STOP_TAG],
                        pad_tag_id=self._pad_idx)

    def forward(self, x: torch.Tensor) -> Tuple[torch.tensor, torch.tensor]:
        masking = x.ne(self._pad_idx)
        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        emissions = self._fc(hiddens)

        score, path = self._crf.decode(emissions, mask=masking.float())

        return score, path

    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        masking = x.ne(self._pad_idx)

        hiddens = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        emissions = self._fc(hiddens)

        nll = self._crf(emissions, y, mask=masking.float())

        return nll