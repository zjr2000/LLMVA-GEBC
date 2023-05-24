from video_llama.common.registry import registry
import torch
import os
from torch.utils.data import DataLoader

@registry.register_runner("runner_feature_extraction")
class RunnerFeatureExtraction:
    
    def __init__(self, cfg, task, model, datasets, job_id):
        self.config = cfg
        self.task = task
        self.dataset = datasets
        self._model = model
        self._job_id = job_id
    
    @property
    def save_dir(self):
        if not hasattr(self, '_save_dir'):
            self._save_dir = self.config.run_cfg.save_dir
            if not os.path.exists(self._save_dir):
                os.mkdir(self._save_dir)
        return self._save_dir
            
    @property
    def data_loader(self):
        if not hasattr(self, '_dataloader'):
            collate_fn = getattr(self.dataset, "collater", None)
            self._dataloader = DataLoader(
                self.dataset,
                num_workers=self.config.run_cfg.num_workers,
                batch_size=self.config.run_cfg.batch_size,
                collate_fn=collate_fn,
            )
        return self._dataloader
    
    @property
    def device(self):
        if not hasattr(self, '_device'):
            self._device = torch.device(self.config.run_cfg.device)

        return self._device
    
    @property
    def model(self):
        # move model to device
        if self._model.device != self.device:
            self._model = self._model.to(self.device)
            self._wrapped_model = self._model

        return self._wrapped_model
    
    def start_extract(self):
        self.task.feature_extraction(self.model, self.data_loader, self.save_dir)