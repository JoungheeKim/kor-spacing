from reformer_pytorch import ReformerLM
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence as pack

class ReformerConfig():

    model_type = "reformer"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        attn_chunks=8,
        num_mem_kv=0,
        full_attn_thres=0,
        reverse_thres=0,
        use_full_attn=False,
        n_hashes=8,
        num_labels=2,
        max_length=512,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.attn_chunks = attn_chunks
        self.num_mem_kv = num_mem_kv
        self.full_attn_thres = full_attn_thres
        self.reverse_thres = reverse_thres
        self.use_full_attn = use_full_attn
        self.n_hashes = n_hashes
        self.num_labels = num_labels
        self.max_length = max_length
        self.pad_token_id = pad_token_id



## https://github.com/lucidrains/reformer-pytorch 참조
class Reformer_TAG(nn.Module):
    def __init__(self, config):
        super().__init__()

        args = ReformerConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_dim,
            intermediate_size=config.hidden_dim*4,
            num_attention_heads=config.num_heads,
            attention_probs_dropout_prob=config.dropout_prob,
            hidden_dropout_prob=config.dropout_prob,
            max_position_embeddings=config.max_token_len,
            max_length=config.max_token_len,
            num_hidden_layers=config.num_layers
        )

        self.config = config
        self.pad_token_id = config.pad_token_id

        num_tokens = args.vocab_size
        dim = args.hidden_size
        depth = args.num_hidden_layers
        max_seq_len = args.max_length
        heads = args.num_attention_heads
        lsh_dropout = args.attention_probs_dropout_prob
        ff_dropout = args.hidden_dropout_prob
        post_attn_dropout = args.attention_probs_dropout_prob
        layer_dropout = args.hidden_dropout_prob
        attn_chunks = args.attn_chunks
        num_mem_kv = args.num_mem_kv
        full_attn_thres = args.full_attn_thres
        reverse_thres = args.reverse_thres
        use_full_attn = args.use_full_attn
        n_hashes = args.n_hashes


        self.reformer = ReformerLM(
            num_tokens=num_tokens,  ## vocab_size
            dim=dim,
            depth=depth,
            max_seq_len=max_seq_len,
            heads=heads,
            lsh_dropout=lsh_dropout,
            ff_dropout=ff_dropout,
            post_attn_dropout=post_attn_dropout,
            layer_dropout=layer_dropout,  # layer dropout from 'Reducing Transformer Depth on Demand' paper
            attn_chunks=attn_chunks,  # process lsh attention in chunks, only way for memory to fit when scaling to 16k tokens
            num_mem_kv=num_mem_kv,  # persistent learned memory key values, from all-attention paper
            full_attn_thres = full_attn_thres, # use full attention if context length is less than set value --> 이거 테스트 해보자
            reverse_thres=reverse_thres, # turn off reversibility for 2x speed for sequence lengths shorter or equal to the designated value  --> 이거 테스트 해보자
            use_full_attn=use_full_attn,
            n_hashes=n_hashes,
            return_embeddings=True
        )

        self.hidden2tag = nn.Linear(config.hidden_dim, config.tag_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.crit = nn.NLLLoss()

    def forward(self, input_ids, lengths, tags=None):
        """ Input has shape [seq length, batch] """
        hiddens = self.reformer(input_ids)
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
        hiddens = self.reformer(input_ids)
        hiddens = self.hidden2tag(hiddens)

        feats = self.softmax(hiddens)
        ps = torch.exp(feats)
        return ps

