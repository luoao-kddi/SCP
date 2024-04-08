import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)


class SelfMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0):
        super(SelfMultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  # 168//4=42
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  #
        self.mlp_key = nn.Linear(emsize, self.all_head_size)  # MLP(168,168)
        self.mlp_query = nn.Linear(emsize, self.all_head_size)
        self.mlp_value = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

        self.mask_unknown = None
        self.former_csz = -1
        self.former_bsz = -1

    # Slice the output of mlpKQV to implement multi-head attention.
    def slice(self, x):
        # [batch_size, context_size, nhead, head_size] or [batch_size, context_size, levelNumK, nhead, head_size]
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def proc_context(self, context):
        # [batch_size, context_size, 4 nhead, 42 head_size]
        context = context.permute(0, 2, 1, 3).contiguous()
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        return context

    def forward(self, embed, embed_unknown, mask):
        '''
            unknown: used to calculate the attention without current node's occupancy
            w/o unkown: former nodes that used to support prediction of current node's occupancy
        '''
        bsz, csz = embed.shape[:2]
        # [batch_size, context_size, all_head_size] -> [batch_size,nhead,context_size,head_size]
        key = self.slice(self.mlp_key(embed))
        key_unknown = self.slice(self.mlp_key(embed_unknown))
        # torch.Size([32, 4, 256, 42])
        query_unkown = self.slice(self.mlp_query(embed_unknown))
        value = self.slice(self.mlp_value(embed))
        value_unknown = self.slice(self.mlp_value(embed_unknown))

        # [batch_size,nhead,context_size,context_size] or [bs,context_size,nhead,levelNumK,levelNumK]
        attn_score = torch.matmul(
            query_unkown, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        attn_score_masked = attn_score + mask[:csz, :csz]
        attn = self.dropout(nn.Softmax(dim=-1)(attn_score_masked))
        output = torch.matmul(attn, value)
        output = self.proc_context(output)
        # torch.Size([32, 4, 256, 256]) ,mask [[0,-inf,-inf,..],[0,0,-inf,...],[0,0,0,...]]

        # current point can only get Level, Octant information, but no Occupancy of itself
        attention_score_zero = torch.sum(query_unkown * key_unknown, dim=3) / math.sqrt(self.head_size)
        if self.mask_unknown is None or self.former_csz != csz or self.former_bsz != bsz:
            self.mask_unknown = torch.eye(csz)[True, True].repeat(bsz, self.nhead, 1, 1).to(embed.device)
            self.former_csz = csz
            self.former_bsz = bsz
        attn_score_unknown = (1 - self.mask_unknown) * attn_score + torch.diag_embed(attention_score_zero)
        attn_score_unknown_masked = attn_score_unknown + mask[:csz, :csz]
        attn_unknown = self.dropout(nn.Softmax(dim=-1)(attn_score_unknown_masked))

        # [batch_size, 4 nhead, context_size, 42 head_size] #torch.Size([32, 4, 256, 42])
        output_unknown = torch.matmul((1 - self.mask_unknown) * attn_unknown, value)
        output_unknown += torch.einsum('ijk,ijkl->ijkl', (torch.diagonal(attn_unknown, dim1=2, dim2=3), value_unknown))
        output_unknown = self.proc_context(output_unknown)
        return output, output_unknown


class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.attn = SelfMultiheadAttention(emsize=ninp, nhead=nhead)
        self.linear1 = nn.Linear(ninp, nhid)
        self.linear2 = nn.Linear(nhid, ninp)
        self.norm1 = nn.LayerNorm(ninp, eps=1e-5)
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src is the integration of leaf node and its ancestors.
    def forward(self, embed, embed_unknown, src_mask):
        _embed, _embed_unknown = self.attn(embed, embed_unknown, src_mask)  # Multi-head Attention
        embed = self.dropout1(_embed) + embed
        embed_unknown = self.dropout1(_embed_unknown) + embed_unknown
        embed = self.norm1(embed)
        embed_unknown = self.norm1(embed_unknown)
        # [batch_size,context_size,ninp] -> [batch_size,context_size,nhid] -> [batch_size,context_size,ninp]
        _embed = self.linear2(self.dropout(torch.relu(self.linear1(embed))))
        _embed_unknown = self.linear2(self.dropout(torch.relu(self.linear1(embed_unknown))))
        embed = embed + self.dropout2(_embed)
        embed_unknown = embed_unknown + self.dropout2(_embed_unknown)
        embed = self.norm2(embed)
        embed_unknown = self.norm2(embed_unknown)
        return embed, embed_unknown


class TransformerModule(nn.Module):

    def __init__(self, cfg):
        super(TransformerModule, self).__init__()
        self.embed_dimension = 4 * (
            cfg.model.occ_embed_dim
            + cfg.model.level_embed_dim
            + cfg.model.octant_embed_dim
            + cfg.model.abs_pos_embed_dim
        )
        self.layers = torch.nn.ModuleList(
            [TransformerLayer(self.embed_dimension,
                              cfg.model.head_num,
                              cfg.model.hidden_dimension,
                              cfg.train.dropout,
                              ) for _ in range(cfg.model.layer_num)])
        self.has_pos_embed = cfg.model.pos_embed
        if self.has_pos_embed:
            self.position_enc = PositionalEncoding(self.embed_dimension, dropout=cfg.train.dropout, max_len=cfg.model.context_size)

    def forward(self, embed, embed_unknown, src_mask):
        if self.has_pos_embed:
            embed = self.position_enc(embed)
            embed_unknown = self.position_enc(embed_unknown)

        for mod in self.layers:
            embed, embed_unknown = mod(embed, embed_unknown, src_mask=src_mask)
        return embed_unknown


class CrossMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0):
        super(CrossMultiheadAttention, self).__init__()
        self.nhead = nhead  # 4
        self.head_size = emsize // nhead  # 168//4=42
        assert self.head_size * nhead == emsize, "embed_dim must be divisible by num_heads"

        self.all_head_size = int(self.nhead * self.head_size)  #
        self.mlp_key = nn.Linear(emsize, self.all_head_size)  # MLP(168,168)
        self.mlp_query = nn.Linear(emsize, self.all_head_size)
        self.mlp_value = nn.Linear(emsize, self.all_head_size)
        self.dropout = nn.Dropout(dropout)

        self.mask_unknown = None
        self.former_csz = -1
        self.former_bsz = -1

    # Slice the output of mlpKQV to implement multi-head attention.
    def slice(self, x):
        # [batch_size, context_size, nhead, head_size] or [batch_size, context_size, levelNumK, nhead, head_size]
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    def proc_context(self, context):
        # [batch_size, context_size, 4 nhead, 42 head_size]
        context = context.permute(0, 2, 1, 3).contiguous()
        context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(*context_shape)
        return context

    def forward(self, embed, q):
        '''
            unknown: used to calculate the attention without current node's occupancy
            w/o unkown: former nodes that used to support prediction of current node's occupancy
        '''
        bsz, csz = embed.shape[:2]
        # [batch_size, context_size, all_head_size] -> [batch_size,nhead,context_size,head_size]
        key = self.slice(self.mlp_key(embed))
        # torch.Size([32, 4, 256, 42])
        query = self.slice(self.mlp_query(q))
        value = self.slice(self.mlp_value(embed))

        # [batch_size,nhead,context_size,context_size] or [bs,context_size,nhead,levelNumK,levelNumK]
        attn_score = torch.matmul(
            query, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        attn_score_masked = attn_score
        attn = self.dropout(nn.Softmax(dim=-1)(attn_score_masked))
        output = torch.matmul(attn, value)
        output = self.proc_context(output)
        # torch.Size([32, 4, 256, 256]) ,mask [[0,-inf,-inf,..],[0,0,-inf,...],[0,0,0,...]]

        # current point can only get Level, Octant information, but no Occupancy of itself
        if self.mask_unknown is None or self.former_csz != csz or self.former_bsz != bsz:
            self.mask_unknown = torch.eye(csz)[True, True].repeat(bsz, self.nhead, 1, 1).to(embed.device)
            self.former_csz = csz
            self.former_bsz = bsz

        # [batch_size, 4 nhead, context_size, 42 head_size] #torch.Size([32, 4, 256, 42])
        return output


class CrossTransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(CrossTransformerLayer, self).__init__()
        self.attn = CrossMultiheadAttention(emsize=ninp, nhead=nhead)
        self.linear1 = nn.Linear(ninp, nhid)
        self.linear2 = nn.Linear(nhid, ninp)
        self.norm1 = nn.LayerNorm(ninp, eps=1e-5)
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src is the integration of leaf node and its ancestors.
    def forward(self, embed, q):
        _embed = self.attn(embed, q)  # Multi-head Attention
        embed = self.dropout1(_embed) + embed
        embed = self.norm1(embed)
        # [batch_size,context_size,ninp] -> [batch_size,context_size,nhid] -> [batch_size,context_size,ninp]
        _embed = self.linear2(self.dropout(torch.relu(self.linear1(embed))))
        embed = embed + self.dropout2(_embed)
        embed = self.norm2(embed)
        return embed


class CrossTransformerModule(nn.Module):

    def __init__(self, cfg):
        super(CrossTransformerModule, self).__init__()
        self.embed_dimension = 4 * (
            cfg.model.occ_embed_dim
            + cfg.model.level_embed_dim
            + cfg.model.octant_embed_dim
            + cfg.model.abs_pos_embed_dim
        )
        self.layers = torch.nn.ModuleList(
            [CrossTransformerLayer(self.embed_dimension,
                              cfg.model.head_num,
                              cfg.model.hidden_dimension,
                              cfg.train.dropout,
                              ) for _ in range(cfg.model.layer_num)])
        self.has_pos_embed = cfg.model.pos_embed
        if self.has_pos_embed:
            self.position_enc = PositionalEncoding(self.embed_dimension, dropout=cfg.train.dropout, max_len=cfg.model.context_size)

    def forward(self, embed, q):
        if self.has_pos_embed:
            embed = self.position_enc(embed)
            q = self.position_enc(q)

        for mod in self.layers:
            embed = mod(embed, q)
        return embed
