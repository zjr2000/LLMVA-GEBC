import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.gebc_dataset import GEBCDataset, EvalGEBCDataset

@registry.register_builder("gebc")
class GEBCBuilder(BaseDatasetBuilder):
    train_dataset_cls = GEBCDataset
    eval_dataset_cls = EvalGEBCDataset
    
    DATASET_CONFIG_DICT = {"default": "configs/datasets/gebc/default.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        # self.build_processors()
        datasets = dict()
        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        split = "train"
        annotations = build_info.annotations
        video_info_path = build_info.video_info_path
        q_former_feature_folder = build_info.q_former_feature_folder
        intern_video_feature_folder = build_info.intern_video_feature_folder
        for split in ['train', 'val', 'test']:
            if split not in ["train", "val", "test"]:
                continue
            is_train = split == "train"
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            annotation_path = annotations.get(split).annotation_path
            datasets[split] = dataset_cls(
                annotation_path=annotation_path,
                video_info_path=video_info_path,
                q_former_feature_folder=q_former_feature_folder,
                intern_video_feature_folder=intern_video_feature_folder
            )
        return datasets