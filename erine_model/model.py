import torch
import torch.nn as nn
from transformers import BertModel
from torch.nn.utils.rnn import pack_padded_sequence as pack
import math
import torch.hub

class ERINE_TAG(nn.Module):

    def __init__(self, config, tokenizer):
        super(ERINE_TAG, self).__init__()

        self.device = config.device
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tag_size = config.tag_size
        self.tanh = nn.Tanh()

        # Maps the output of the LSTM into tag space.

        word_vec_dim = tokenizer.word_vec_dim
        word_vocab_size = tokenizer.word_vocab_size
        self.emb = nn.Embedding(word_vocab_size, word_vec_dim)

        layers = []
        layers.append(nn.LayerNorm(word_vec_dim, eps=config.layer_norm_eps))
        layers.append(nn.Dropout(config.dropout_prob))
        for _ in range(config.num_layers):
            layers.append(nn.Linear(word_vec_dim, word_vec_dim))
            layers.append(torch.nn.ReLU())

        self.linear_layer = nn.Sequential(*layers)

        self.project = nn.Linear(self.bert.config.hidden_size, word_vec_dim)
        self.hidden2tag = nn.Linear(word_vec_dim, self.tag_size)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.crit = nn.NLLLoss()

        self._set_embedding(tokenizer.embedding)

    def _set_embedding(self, embedding):
        self.emb.weight.data.copy_(embedding)
        return True

    def forward(self, input_ids, word_input_ids, lengths, tags=None):
        
        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.project(hiddens)
        hiddens = self.tanh(hiddens)

        word_hiddens = self.emb(word_input_ids)
        word_hiddens = self.linear_layer(word_hiddens)

        hiddens = hiddens + word_hiddens
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

    def get_prob(self, input_ids, word_input_ids, lengths):
        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.tanh(self.fc(hiddens))
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps


class ERINE_AGG(nn.Module):

    def __init__(self, config, tokenizer):
        super(ERINE_AGG, self).__init__()

        self.device = config.device
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.tag_size = config.tag_size
        self.tanh = nn.Tanh()

        # Maps the output of the LSTM into tag space.

        word_vec_dim = tokenizer.word_vec_dim
        word_intermediate_size = word_vec_dim*4
        word_vocab_size = tokenizer.word_vocab_size
        self.emb = nn.Embedding(word_vocab_size, word_vec_dim)

        layers = []
        layers.append(nn.LayerNorm(word_vec_dim, eps=config.layer_norm_eps))
        layers.append(nn.Dropout(config.dropout_prob))



        for _ in range(config.num_layers):
            layers.append(WordSelfAttention(hidden_size=word_vec_dim,
                                            num_attention_heads=config.num_heads,
                                            attention_probs_dropout_prob=config.dropout_prob))
            layers.append(nn.Linear(word_vec_dim, word_intermediate_size))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(word_intermediate_size, word_vec_dim))


        self.linear_layer = nn.Sequential(*layers)


        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(config.num_layers):
            self.attentions.append(WordSelfAttention(hidden_size=word_vec_dim,
                                            num_attention_heads=config.num_heads,
                                            attention_probs_dropout_prob=config.dropout_prob))
            self.feed_forwards.append(nn.Sequential(nn.Linear(word_vec_dim, word_intermediate_size),
                                                    nn.ReLU(),
                                                    nn.Linear(word_intermediate_size, word_vec_dim)))
            self.layer_norms_1.append(nn.LayerNorm(word_vec_dim, eps=config.layer_norm_eps))
            self.layer_norms_2.append(nn.LayerNorm(word_vec_dim, eps=config.layer_norm_eps))


        self.project = nn.Linear(self.bert.config.hidden_size, word_vec_dim)
        self.hidden2tag = nn.Linear(word_vec_dim, self.tag_size)

        self.softmax = nn.LogSoftmax(dim=-1)
        self.crit = nn.NLLLoss()

        self.freeze = config.freeze
        self._set_embedding(tokenizer.embedding)
        self.dropout = nn.Dropout(config.dropout_prob)


    def _set_embedding(self, embedding):
        self.emb.weight.data.copy_(embedding)
        if self.freeze is not None:
            self.emb.weight.requires_grad = False
            return False
        return True

    def forward(self, input_ids, word_input_ids, lengths, tags=None):

        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.project(hiddens)
        hiddens = self.tanh(hiddens)
        word_hiddens = self.emb(word_input_ids)
        h = self.linear_layer(word_hiddens)

        ## Aggregator
        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):

            x = attention(h)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_1(h)

            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h
            h = layer_norm_2(h)

        hiddens = hiddens + h
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

    def get_prob(self, input_ids, word_input_ids, lengths):
        output = self.bert(input_ids)
        hiddens = output[0]
        hiddens = self.tanh(self.fc(hiddens))
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps


class WordSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob=0, output_attentions=False):
        super().__init__()
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else context_layer
        return outputs




