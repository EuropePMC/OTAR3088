from abc import ABC, abstractmethod



class PrepareDataset(ABC):
    def __init__(self, cfg, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run

    @abstractmethod
    def prepare(self):
        pass




class DatasetLoader(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        
    @abstractmethod
    def load(self):
        pass