from abc import ABC, abstractmethod



class PrepareDataset(ABC):
    def __init__(self, cfg, wandb_run=None):
        self.cfg = cfg
        self.wandb_run = wandb_run

    @abstractmethod
    def prepare(self):
        pass




class DatasetLoader(ABC):
    def __init__(self, cfg, hf_cache_dir=None):
        self.cfg = cfg
        self.hf_cache_dir = hf_cache_dir
        
    @abstractmethod
    def load(self):
        pass