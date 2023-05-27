import json
import pickle

TYPE_MAPPING = {
    'subject': 'Subject',
    'status_before': 'Status_Before',
    'status_after': 'Status_After'
}

groundtruth_file_path = 'data/annotations_origin/test_timestamp.json'
prediction_file_path = 'video_llama/output/video_blip2_opt_highest_f1_12frame_eval/20230527160/result/test_epochbest.json'
save_file_path = 'submit_test.pkl'

with open(groundtruth_file_path, 'r') as f:
    groundtruths = json.load(f)
    
with open(prediction_file_path, 'r') as f:
    predictions = json.load(f)
    
all_boundary_keys = []
for key, video_data in groundtruths.items():
    boundary_ids = [val['boundary_id'] for val in video_data]
    all_boundary_keys.extend(boundary_ids)
all_boundary_keys = set(all_boundary_keys)

new_predictions = {}
for pred in predictions:
    boundary_id = pred['boundary_id']
    caption = pred['caption']
    caption_type = pred['type']
    caption_type = TYPE_MAPPING[caption_type]
    if boundary_id not in new_predictions:
        new_predictions[boundary_id] = {}
    new_predictions[boundary_id][caption_type] = caption
print(len(all_boundary_keys))
print(len(new_predictions.keys()))    
PADDING_DATA = {'Subject': '', 'Status_Before': '', 'Status_After': ''}
for key in all_boundary_keys:
    if key not in new_predictions:
        print('Missing {} in predictions, add padding'.format(key))
        new_predictions[key] = PADDING_DATA
        
with open(save_file_path, 'wb') as f:
    pickle.dump(new_predictions, f)
    
     

