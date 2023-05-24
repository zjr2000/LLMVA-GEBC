"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
import pandas as pd
import decord
from decord import VideoReader
import random
import torch


        
class GEBCVideoDataset(BaseDataset):
    def __init__(self, vis_processor, vis_root):
        """
        vis_root (string): Root directory of videos (e.g. coco/images/)
        """
        super().__init__(vis_processor, text_processor=None, vis_root=vis_root, ann_paths=[])
        video_paths = os.listdir(vis_root)
        self.annotation = [{'video_id': video_name[0:11], 'video': video_name} for video_name in video_paths]

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "video_id": ann["video_id"],
        }