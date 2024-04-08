import math
import torch
from torch import nn
import pytorch_lightning as pl

from models.attention_model import TransformerModule


class OctAttention(pl.LightningModule):
    def __init__(self, cfg):
        super(OctAttention, self).__init__()
        self.cfg = cfg

        self.transformer_encoder = TransformerModule(cfg)

        self.occ_enc = nn.Embedding(cfg.model.token_num + 1, cfg.model.occ_embed_dim)

        self.level_enc = nn.Embedding(
            cfg.model.max_octree_level + 1, cfg.model.level_embed_dim
        )
        self.octant_enc = nn.Embedding(9, cfg.model.octant_embed_dim)

        self.abs_pos_embed_dim = cfg.model.abs_pos_embed_dim
        if self.abs_pos_embed_dim:
            self.abs_pos_enc = nn.Linear(3, self.abs_pos_embed_dim)
        self.act = nn.ReLU()

        self.embed_dimension = 4 * (
            cfg.model.occ_embed_dim
            + cfg.model.level_embed_dim
            + cfg.model.octant_embed_dim
            + self.abs_pos_embed_dim
        )
        self.decoder0 = nn.Linear(self.embed_dimension, self.embed_dimension)
        self.decoder1 = nn.Linear(self.embed_dimension, cfg.model.token_num)

        self.criterion = nn.CrossEntropyLoss()
        mask = (
            torch.triu(torch.ones(cfg.model.context_size, cfg.model.context_size)) == 1
        ).transpose(0, 1)
        self.register_buffer(
            "mask",
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0)),
        )

    def forward(self, data, pos=None):
        bsz = data.shape[0]
        csz = data.shape[1]

        occupancy = data[:, :, :, 0]  # [0~255] 255 for padding data
        level = data[:, :, :, 1]  # [0~12] 0 for padding data
        octant = data[:, :, :, 2]  # [0~8] 0 for padding data

        # the max level in traning dataset is 10 for MPEG/MVUB 12 for KITTI
        if self.cfg.train.type == 'obj':
            level -= torch.clip(level[:, :, -1:] - 10, 0, None)
        else:
            level -= torch.clip(level[:, :, -1:] - 12, 0, None)
        torch.clip_(level, 0, self.cfg.model.max_octree_level)

        occ_embed = self.occ_enc(occupancy)
        # unknown occupancy set to 255 (normal ones are 0~244)
        unknown_occ = torch.ones_like(occupancy[:, :, -1]) * 255
        unknown_occ_embed = torch.zeros_like(occ_embed)
        # [:-1] is ancients, which are known
        unknown_occ_embed[:, :, :-1] = occ_embed[:, :, :-1]
        unknown_occ_embed[:, :, -1] = self.occ_enc(unknown_occ)

        level_embed = self.level_enc(level)
        octant_embed = self.octant_enc(octant)

        abs_pos_embed = None
        if self.abs_pos_embed_dim:
            abs_pos_embed = self.abs_pos_enc(pos)

        embed = self.cat_embeds(bsz, csz, occ_embed, level_embed, octant_embed, abs_pos_embed)
        embed_unknown = self.cat_embeds(bsz, csz, unknown_occ_embed, level_embed, octant_embed, abs_pos_embed)

        output = self.transformer_encoder(embed, embed_unknown, self.mask)
        output = self.decoder1(self.act(self.decoder0(output)))
        return output

    def cat_embeds(
        self, bsz, csz, occ_embed, level_embed, octant_embed, abs_pos_embed
    ):
        if abs_pos_embed is not None:
            embed = torch.cat(
                (occ_embed, level_embed, octant_embed, abs_pos_embed), 3
            ).reshape((bsz, csz, self.embed_dimension)) * math.sqrt(
                self.embed_dimension
            )
        else:
            embed = torch.cat((occ_embed, level_embed, octant_embed), 3).reshape(
                (bsz, csz, self.embed_dimension)
            ) * math.sqrt(self.embed_dimension)

        return embed

    def configure_optimizers(self):
        optim_cfg = self.cfg.train.optimizer
        sched_cfg = self.cfg.train.lr_scheduler
        if optim_cfg.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
        else:
            raise NotImplementedError()
        if sched_cfg.name == "StepLR":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=sched_cfg.step_size, gamma=sched_cfg.gamma
            )
        else:
            raise NotImplementedError()

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch):
        data, pos, labels = batch
        pred = self(data, pos)
        loss = self.criterion(
            pred.view(-1, self.cfg.model.token_num), labels.reshape(-1)
        ) / math.log(2)
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss
