import json


original_annotation_path = 'data/annotations_origin/train_all_annotation.json'
video_info_path = 'data/annotations/video_info.json'
target_annotation_path = 'data/annotations/train_all_annotation.json'

with open(video_info_path, 'r') as f:
    video_info = json.load(f)

with open(original_annotation_path, 'r') as f:
    ori_annotations = json.load(f)
    
target_data = {}


for video_key, all_annotations in ori_annotations.items():
    cnt = 0
    real_key = video_key[0:11]
    if real_key not in video_info:
        continue
    boundary_list = []
    duration = video_info[real_key]
    for one_person_annotations in all_annotations:
        prev_timestamp = 0        
        for idx, boundary in enumerate(one_person_annotations['boundary_list']):
            next_timestamp = one_person_annotations['boundary_list'][idx + 1]['start_time'] if idx + 1 < len(one_person_annotations['boundary_list']) else duration
            new_boundary_data = {
                'boundary_id': real_key + '_' + str(cnt),
                'prev_timestamp': prev_timestamp,
                'timestamp': boundary['start_time'],
                'next_timestamp': next_timestamp,
                'label': boundary['label'],
                'subject': boundary['subject'],
                'status_before': boundary['status_before'],
                'status_after': boundary['status_after']
            }
            boundary_list.append(new_boundary_data)
            cnt += 1
            prev_timestamp = boundary['start_time']
    target_data[real_key] = boundary_list
    
with open(target_annotation_path, 'w') as f:
    json.dump(target_data, f)