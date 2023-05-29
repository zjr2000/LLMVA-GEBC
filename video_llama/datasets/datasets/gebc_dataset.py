import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
import json
import numpy as np
import pickle
import torch
from torch.nn.utils.rnn import pad_sequence

def read_file(path, MEAN=0., VAR=1., data_norm=False):
    if os.path.exists(path):
        ext = path.split('.')[-1]
        if ext == 'npy':
            feats = np.load(path)
        elif ext == 'pkl':
            with open(path, 'rb') as f:
                feats = pickle.load(f)
        else:
            raise NotImplementedError

        padding = False
    else:
        raise FileNotFoundError('{} not exists'.format(path))
    if data_norm:
        feats = (feats - MEAN) / np.sqrt(VAR)
    return feats, padding


def get_feats(key, vf_type, vf_folder, data_norm=False):
    MEAN = VAR = 0
    if vf_type == 'q_former_tokens':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.npy')
    elif vf_type == 'intern_video_feature':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    else:
        raise AssertionError('feature type error: {}'.format(vf_type))
    feats, padding = read_file(path, MEAN, VAR, data_norm)

    assert feats.shape[-1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats


def build_prompt(boundary_type, caption_type):
    prompt = 'This video describes the {}.'.format(boundary_type.lower())
    if caption_type == 'subject':
        prompt += 'The subject is'
    elif caption_type == 'status_before':
        prompt += 'Status before change is'
    else:
        prompt += 'Status after change is'
    return prompt


class GEBCDataset(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, intern_video_feature_folder):
        self.q_former_feature_folder = q_former_feature_folder
        self.intern_video_feature_folder = intern_video_feature_folder
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        with open(video_info_path, 'r') as f:
            self.video_info = json.load(f)
        self._load_annotations(annotation_path)
            

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_boundaries in data.items():
            if not key in self.video_info:
                print('missing key:', key)
                continue
            duration = self.video_info[key]
            for video_anno in video_boundaries:
                boundary_duration = float(video_anno['next_timestamp']) - float(video_anno['prev_timestamp'])
                subject_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'subject',
                    'caption': video_anno['subject']
                }
                status_before_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'status_before',
                    'caption': video_anno['status_before']
                }
                status_after_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'status_after',
                    'caption': video_anno['status_after']
                }
                self.annotation.append(subject_data)
                self.annotation.append(status_before_data)
                self.annotation.append(status_after_data)
    
    
    def __getitem__(self, index):
        item_data = self.annotation[index]
        # Prepare boundary information
        boundary_timestamp, boundary_duration, duration = item_data['timestamp'], item_data['boundary_duration'], item_data['duration']
        reference_point = np.array([boundary_timestamp / duration, boundary_duration / duration])
        reference_point = torch.from_numpy(reference_point)
        # Prepare prompt
        boundary_type, caption_type = item_data['label'], item_data['type']
        prompt = build_prompt(boundary_type, caption_type)
        # Load caption
        caption = item_data['caption']
        # Load feature
        q_former_tokens = get_feats(item_data['boundary_id'], 'q_former_tokens', self.q_former_feature_folder)
        q_former_tokens = torch.from_numpy(q_former_tokens)
        # load intern video feature
        intern_video_feature = get_feats(item_data['boundary_id'], 'intern_video_feature', self.intern_video_feature_folder)
        intern_video_feature = torch.from_numpy(intern_video_feature)
        return {
            'image_query_tokens': q_former_tokens,
            'intern_video_feature': intern_video_feature,
            'reference_points': reference_point,
            'prompt': prompt,
            'text_input': caption,
            'boundary_id': item_data['boundary_id']
        }
    
        
    def collater(self, samples):   
        q_former_tokens = torch.stack([sample['image_query_tokens'] for sample in samples], 0)
        intern_video_feature = pad_sequence([sample['intern_video_feature'].unsqueeze(1) for sample in samples], 
                                    batch_first=True, padding_value=0)
        reference_points = torch.stack([sample['reference_points'] for sample in samples], 0)
        prompt = [sample['prompt'] for sample in samples]
        text_input = [sample['text_input'] for sample in samples]
        boundary_ids = [sample['boundary_id'] for sample in samples]
        return {
            'image_query_tokens': q_former_tokens,
            'intern_video_feature': intern_video_feature,
            'reference_points': reference_points,
            'prompt': prompt,
            'text_input': text_input,
            'boundary_id': boundary_ids
        }
        
        
class EvalGEBCDataset(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, intern_video_feature_folder):
        self.q_former_feature_folder = q_former_feature_folder
        self.intern_video_feature_folder = intern_video_feature_folder
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        with open(video_info_path, 'r') as f:
            self.video_info = json.load(f)
        self._load_annotations(annotation_path)
            

    def _load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        for key, video_boundaries in data.items():
            if not key in self.video_info:
                print('missing key:', key)
                continue
            duration = self.video_info[key]
            for video_anno in video_boundaries:
                boundary_duration = float(video_anno['next_timestamp']) - float(video_anno['prev_timestamp'])
                subject_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'subject',
                }
                status_before_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'status_before',
                }
                status_after_data = {
                    'boundary_id': video_anno['boundary_id'],
                    'timestamp': video_anno['timestamp'],
                    'label': video_anno['label'],
                    'duration': duration,
                    'boundary_duration': boundary_duration,
                    'type': 'status_after',
                }
                self.annotation.append(subject_data)
                self.annotation.append(status_before_data)
                self.annotation.append(status_after_data)
    
    
    def __getitem__(self, index):
        item_data = self.annotation[index]
        # Prepare boundary information
        boundary_timestamp, boundary_duration, duration = item_data['timestamp'], item_data['boundary_duration'], item_data['duration']
        reference_point = np.array([boundary_timestamp / duration, boundary_duration / duration])
        reference_point = torch.from_numpy(reference_point)
        # Prepare prompt
        boundary_type, caption_type = item_data['label'], item_data['type']
        prompt = build_prompt(boundary_type, caption_type)
        # Load feature
        q_former_tokens = get_feats(item_data['boundary_id'], 'q_former_tokens', self.q_former_feature_folder)
        q_former_tokens = torch.from_numpy(q_former_tokens)
        # load intern video feature
        intern_video_feature = get_feats(item_data['boundary_id'], 'intern_video_feature', self.intern_video_feature_folder)
        intern_video_feature = torch.from_numpy(intern_video_feature)
        return {
            'image_query_tokens': q_former_tokens,
            'intern_video_feature': intern_video_feature,
            'reference_points': reference_point,
            'prompt': prompt,
            'boundary_id': item_data['boundary_id'],
            'caption_type': caption_type
        }
    
        
    def collater(self, samples):   
        q_former_tokens = torch.stack([sample['image_query_tokens'] for sample in samples], 0)
        intern_video_feature = pad_sequence([sample['intern_video_feature'].unsqueeze(1) for sample in samples], 
                                    batch_first=True, padding_value=0)
        reference_points = torch.stack([sample['reference_points'] for sample in samples], 0)
        prompt = [sample['prompt'] for sample in samples]
        boundary_ids = [sample['boundary_id'] for sample in samples]
        caption_types = [sample['caption_type'] for sample in samples]
        return {
            'image_query_tokens': q_former_tokens,
            'intern_video_feature': intern_video_feature,
            'reference_points': reference_points,
            'prompt': prompt,
            'boundary_id': boundary_ids,
            'caption_type': caption_types
        }