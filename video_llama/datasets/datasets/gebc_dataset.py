import os
from video_llama.datasets.datasets.base_dataset import BaseDataset
import json
import numpy as np
import pickle
import torch
from scipy.interpolate import interp1d

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
    elif vf_type == 'intern_video':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'omni':
        feat_dim = 1536
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    elif vf_type == 'clip':
        feat_dim = 768
        path = os.path.join(vf_folder, key[0:11] + '.pkl')
    else:
        raise AssertionError('feature type error: {}'.format(vf_type))
    feats, padding = read_file(path, MEAN, VAR, data_norm)

    assert feats.shape[-1] == feat_dim, 'load {} error, got shape {}'.format(path, feats.shape)
    return feats

def resizeFeature(inputData, newSize, sample_method):
    # inputX: (temporal_length,feature_dimension) #
    originalSize = len(inputData)
    # print originalSize
    if originalSize == 1:
        inputData = np.reshape(inputData, [-1])
        return np.stack([inputData] * newSize)
    x = np.array(range(originalSize))
    f = interp1d(x, inputData, axis=0, kind=sample_method)
    x_new = [i * float(originalSize - 1) / (newSize - 1) for i in range(newSize)]
    y_new = f(x_new)
    return y_new

def build_prompt(boundary_type, caption_type):
    prompt = 'This video describes the {}.'.format(boundary_type.lower())
    if caption_type == 'subject':
        prompt += 'The subject is'
    elif caption_type == 'status_before':
        prompt += 'Status before change is'
    else:
        prompt += 'Status after change is'
    return prompt


def load_object_feats(key, object_feature_path_map, num_frame=12, max_obj_num=10, feat_dim=2054):
    frame_num = len(object_feature_path_map[key])
    frame_indicies = list(range(frame_num))
    if len(frame_indicies) < num_frame:
        last_idx = frame_indicies[-1]
        while len(frame_indicies) < num_frame:
            frame_indicies.append(last_idx)
    step = len(frame_indicies) // num_frame
    frame_indicies = [frame_indicies[i * step] for i in range(num_frame)]
    object_features = []
    for frame_idx in frame_indicies:
        frame_obj_feature_path = object_feature_path_map[key][frame_idx]
        feature, _  = read_file(frame_obj_feature_path)
        if feature.shape[0] < max_obj_num:
            feature = np.concatenate([feature, np.zeros((max_obj_num, feat_dim))])
        feature = feature[0:max_obj_num,:]
        object_features.append(feature)
    object_features = np.stack(object_features, axis=0)
    object_features = torch.from_numpy(object_features)
    return object_features


class GEBCDataset(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, other_feature_names, other_feature_folders, max_seq_len, object_feature_path_map):
        self.q_former_feature_folder = q_former_feature_folder
        self.other_feature_names = other_feature_names
        self.other_feature_folders = other_feature_folders
        self.max_seq_len = max_seq_len
        with open(object_feature_path_map, 'r') as f:
            self.object_feature_path_map = json.load(f)
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
        q_former_tokens = torch.from_numpy(q_former_tokens) # (t,q,h)
        # load  other feature
        other_features_list = [] # (a,t,h), a is the number of other features
        for i, folder in enumerate(self.other_feature_folders):
            other_feature = get_feats(item_data['boundary_id'], self.other_feature_names[i], folder) # (t,h)
            other_features_list.append(other_feature)
        object_features = load_object_feats(item_data['boundary_id'][0:11], self.object_feature_path_map, num_frame=12, max_obj_num=10, feat_dim=2054)
        
        return {
            'image_query_tokens': q_former_tokens,
            'other_features_list': other_features_list,
            'reference_points': reference_point,
            'prompt': prompt,
            'text_input': caption,
            'boundary_id': item_data['boundary_id'],
            'object_features': object_features
        }
    
        
    def collater(self, samples):   
        q_former_tokens = torch.stack([sample['image_query_tokens'] for sample in samples], 0) # (b,t,q,h)
        
        other_features_list = [sample['other_features_list'] for sample in samples] # (b,a,t,h)
        batch_size = len(other_features_list)
        n_feature= len(other_features_list[0])
        for i in range(batch_size):
            for j in range(n_feature):
                other_features_list[i][j] = resizeFeature(other_features_list[i][j], self.max_seq_len, 'nearest') #(t,h)
                other_features_list[i][j] = torch.from_numpy(other_features_list[i][j]).unsqueeze(1) # (t,q,h)
            
            other_features_list[i] = torch.cat(other_features_list[i], dim=-1) # (t,q,h1)

        other_features_list = torch.stack(other_features_list, dim=0) # (b,t,q,h1)

        reference_points = torch.stack([sample['reference_points'] for sample in samples], 0)
        object_features = torch.stack([sample['object_features'] for sample in samples], 0)
        prompt = [sample['prompt'] for sample in samples]
        text_input = [sample['text_input'] for sample in samples]
        boundary_ids = [sample['boundary_id'] for sample in samples]
        return {
            'image_query_tokens': q_former_tokens,
            'other_features_list': other_features_list,
            'reference_points': reference_points,
            'prompt': prompt,
            'text_input': text_input,
            'object_features': object_features,
            'boundary_id': boundary_ids
        }
        
        
class EvalGEBCDataset(BaseDataset):
    def __init__(self, annotation_path, video_info_path, q_former_feature_folder, other_feature_names, other_feature_folders, max_seq_len, object_feature_path_map):
        self.q_former_feature_folder = q_former_feature_folder
        self.other_feature_names = other_feature_names
        self.other_feature_folders = other_feature_folders
        self.max_seq_len = max_seq_len
        super().__init__(vis_processor=None, text_processor=None, vis_root=None, ann_paths=[])
        with open(video_info_path, 'r') as f:
            self.video_info = json.load(f)
        self._load_annotations(annotation_path)
        with open(object_feature_path_map, 'r') as f:
            self.object_feature_path_map = json.load(f)
            

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
        # load  other feature
        other_features_list = [] # (a,t,h), a is the number of other features
        for i, folder in enumerate(self.other_feature_folders):
            other_feature = get_feats(item_data['boundary_id'], self.other_feature_names[i], folder) # (t,h)
            other_features_list.append(other_feature)
        object_features = load_object_feats(item_data['boundary_id'][0:11], self.object_feature_path_map, num_frame=12, max_obj_num=6, feat_dim=2054)
        return {
            'image_query_tokens': q_former_tokens,
            'other_features_list': other_features_list,
            'reference_points': reference_point,
            'prompt': prompt,
            'boundary_id': item_data['boundary_id'],
            'object_features': object_features,
            'caption_type': caption_type
        }
    
        
    def collater(self, samples):   
        q_former_tokens = torch.stack([sample['image_query_tokens'] for sample in samples], 0)
        
        other_features_list = [sample['other_features_list'] for sample in samples] # (b,a,t,h)
        batch_size = len(other_features_list)
        n_feature= len(other_features_list[0])
        for i in range(batch_size):
            for j in range(n_feature):
                other_features_list[i][j] = resizeFeature(other_features_list[i][j], self.max_seq_len, 'nearest') #(t,h)
                other_features_list[i][j] = torch.from_numpy(other_features_list[i][j]).unsqueeze(1) # (t,q,h)
            
            other_features_list[i] = torch.cat(other_features_list[i], dim=-1) # (t,q,h1)

        other_features_list = torch.stack(other_features_list, dim=0) # (b,t,q,h1)

        reference_points = torch.stack([sample['reference_points'] for sample in samples], 0)
        prompt = [sample['prompt'] for sample in samples]
        boundary_ids = [sample['boundary_id'] for sample in samples]
        caption_types = [sample['caption_type'] for sample in samples]
        object_features = torch.stack([sample['object_features'] for sample in samples], 0)
        return {
            'image_query_tokens': q_former_tokens,
            'other_features_list': other_features_list,
            'reference_points': reference_points,
            'prompt': prompt,
            'boundary_id': boundary_ids,
            'object_features': object_features,
            'caption_type': caption_types
        }