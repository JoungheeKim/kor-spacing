import torch
import torch.nn as nn
from transformers.modeling_bert import BertSelfAttention, BertLayerNorm


class MyEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(MyEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# This all implementations referred from huggingface
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embeddings = MyEmbeddings(config)

        intermediate_size = config.hidden_size * 4

        self.causal = False
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.attentions, self.feed_forwards = nn.ModuleList(), nn.ModuleList()
        self.layer_norms_1, self.layer_norms_2 = nn.ModuleList(), nn.ModuleList()

        for _ in range(config.num_layers):
            self.attentions.append(BertSelfAttention(config))
            self.feed_forwards.append(nn.Sequential(nn.Linear(config.hidden_size, intermediate_size),
                                                    nn.ReLU(),
                                                    nn.Linear(intermediate_size, config.hidden_size)))
            self.layer_norms_1.append(nn.LayerNorm(config.hidden_size, eps=1e-12))
            self.layer_norms_2.append(nn.LayerNorm(config.hidden_size, eps=1e-12))

    def forward(self, x, padding_mask=None):
        """ x has shape [seq length, batch], padding_mask has shape [batch, seq length] """
        positions = torch.arange(len(x), device=x.device).unsqueeze(-1)
        h = self.embeddings(x)

        for layer_norm_1, attention, layer_norm_2, feed_forward in zip(self.layer_norms_1, self.attentions,
                                                                       self.layer_norms_2, self.feed_forwards):
            h = layer_norm_1(h)
            x, _ = attention(h)
            x = self.dropout(x)
            h = x + h

            h = layer_norm_2(h)
            x = feed_forward(h)
            x = self.dropout(x)
            h = x + h

        return h



class TransformerClassification(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_class = config.num_class

        self.pad_token_id = config.pad_token_id

        ## Transformer
        self.transformer = Transformer(config)

        ## Output
        self.classification_head = nn.Linear(config.hidden_size, config.num_class)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, input_ids, clf_tokens_mask, clf_labels=None):

        #pad_tokens_mask = input_ids != self.pad_token_id

        """ Input has shape [seq length, batch] """
        hidden_states = self.transformer(input_ids)
        clf_tokens_states = (hidden_states * clf_tokens_mask.unsqueeze(-1)).sum(dim=1)
        clf_logits = self.classification_head(clf_tokens_states)

        if clf_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(clf_logits, clf_labels)
            return self.softmax(clf_logits), loss

        return self.softmax(clf_logits)


    def get_prob(self, input_ids, length):
        hiddens = self.transformer(input_ids)
        hiddens = self.tanh(self.fc(hiddens))
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps

