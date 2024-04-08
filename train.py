import os
import hydra
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from models import *
from dataloaders.oct_attn_dataloader import OctAttnLoader
from pytorch_lightning.loggers import WandbLogger


@hydra.main(version_base=None, config_path="configs", config_name="train_obj.yaml")
def main(cfg):
    print(cfg)

    # meta
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpus).replace(' ', '')[1:-1]

    # random seed
    seed_everything(cfg.train.seed, workers=True)

    # set model
    model_name = cfg.model.class_name
    model_class = eval(model_name)

    if cfg.train.load_pretrain:
        # model = model_class.load_from_checkpoint(cfg.train.load_ckpt, cfg=cfg, strict=False)
        model = model_class(cfg)
        missing_keys, unexpected_keys = model.load_pretrain(cfg.train.load_pretrain, strict=False)
        print(missing_keys)
        print(unexpected_keys)
    else:
        model = model_class(cfg)

    # set data
    dataloader = OctAttnLoader(cfg.data)

    # train
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    save_path = hydra_cfg['runtime']['output_dir'] + '/ckpt'
    print('saving in', save_path[save_path.find('outputs'):-5])

    trainer = Trainer(max_epochs=cfg.train.epoch,
                    default_root_dir=save_path,
                    deterministic=True,
                    accelerator='gpu',
                    devices=len(cfg.gpus),
                    precision='bf16',
                    logger=WandbLogger(project='wandb_output'),
                    callbacks=[
                        ModelCheckpoint(save_path, save_top_k=-1),
                    ],
    )
    trainer.fit(model, dataloader)


if __name__ == '__main__':
    main()
