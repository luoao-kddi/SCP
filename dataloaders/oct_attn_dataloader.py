from dataloaders.oct_attn_dataset import OctAttnDataset
from dataloaders.ehem_dataset import EHEMDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class OctAttnLoader(LightningDataModule):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_set = None

    def prepare_data(self):
        pass

    def setup(self, stage):
        if not self.train_set:
            if self.cfg.dataset_name == "OctAttnDataset":
                self.train_set = OctAttnDataset(self.cfg)
            elif self.cfg.dataset_name == "EHEM":
                self.train_set = EHEMDataset(self.cfg)
            else:
                raise NotImplementedError(f"Not Implemented Dataset: {self.cfg.dataset_name}")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers, pin_memory=self.cfg.pin_memory, drop_last=True)
