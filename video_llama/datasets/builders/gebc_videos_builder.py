
from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.gebc_video_dataset import GEBCVideoDataset

@registry.register_builder("gebc_videos")
class GEBCVideosBuilder(BaseDatasetBuilder):
    train_dataset_cls = GEBCVideoDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/gebc_videos/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass
    
    def build_processors(self):
        vis_proc_cfg = self.config.get("vis_processor")

        if vis_proc_cfg is not None:
            vis_eval_cfg = vis_proc_cfg.get("eval")

            self.vis_processors["eval"] = self._build_proc_from_cfg(vis_eval_cfg)

    
    def build(self):
        self.build_processors()

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        dataset = dataset_cls(
            vis_processor=self.vis_processors['eval'],
            vis_root=build_info.videos_dir,
        )

        return dataset