from cv2 import normalize
import torch
from torchvision.transforms import transforms
from torchvision.io import read_video
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from TSP.extract_features.eval_video_dataset import EvalVideoDataset
import torchvision
import pandas as pd
import argparse
from tqdm import tqdm
from einops import rearrange

class ToNormalizedFloatTensor(object):
    def __call__(self, vframes):
        vframes = rearrange(vframes, 'T H W C -> T C H W')
        vframes = vframes.to(torch.float32) / 255
        return vframes


def evaluate(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            clip = sample['clip'].to(device, non_blocking=True).squeeze(1)
            feat = model(clip)
            data_loader.dataset.save_features(feat, sample)


def main(args):
    print(args)
    print('TORCH VERSION: ', torch.__version__)
    print('TORCHVISION VERSION: ', torchvision.__version__)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # model, transform = clip.load(args.clip_backbone, device=device)

    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    model.to('cuda')
    # cropping = transform.transforms[:2]
    # normalize = transform.transforms[-1]
    # reconstruct_transforms = [ToNormalizedFloatTensor()]
    # reconstruct_transforms.extend(cropping)
    # reconstruct_transforms.append(normalize)
    # transform = transforms.Compose(reconstruct_transforms)
    transform =  transforms.Compose([
                            ToNormalizedFloatTensor(),
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            # transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])
                        ])
    

    metadata_df = pd.read_csv(args.metadata_csv_filename)
    shards = np.linspace(0,len(metadata_df),args.num_shards+1).astype(int)
    start_idx, end_idx = shards[args.shard_id], shards[args.shard_id+1]
    print(f'shard-id: {args.shard_id + 1} out of {args.num_shards}, '
        f'total number of videos: {len(metadata_df)}, shard size {end_idx-start_idx} videos')

    metadata_df = metadata_df.iloc[start_idx:end_idx].reset_index()
    metadata_df['is-computed-already'] = metadata_df['filename'].map(lambda f:
        os.path.exists(os.path.join(args.output_dir, os.path.basename(f).split('.')[0] + '.pkl')))
    metadata_df = metadata_df[metadata_df['is-computed-already']==False].reset_index(drop=True)
    print(f'Number of videos to process after excluding the ones already computed on disk: {len(metadata_df)}')

    dataset = EvalVideoDataset(
        metadata_df=metadata_df,
        root_dir=args.data_path,
        clip_length=1,
        frame_rate=args.frame_rate,
        stride=args.stride,
        output_dir=args.output_dir,
        transforms=transform)

    print('CREATING DATA LOADER')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    

    print('START FEATURE EXTRACTION')
    evaluate(model, data_loader, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Features extraction script')

    parser.add_argument('--data-path', required=True,
                        help='Path to the directory containing the videos files')
    parser.add_argument('--metadata-csv-filename', required=True,
                        help='Path to the metadata CSV file')

    # parser.add_argument('--clip_backbone', required=True,
    #                     help='Clip backbone')

    parser.add_argument('--frame-rate', default=15, type=int,
                        help='Frames-per-second rate at which the videos are sampled (default: 15)')
    parser.add_argument('--stride', default=16, type=int,
                        help='Number of frames (after resampling with frame-rate) between consecutive clips (default: 16)')

    parser.add_argument('--device', default='cuda',
                        help='Device to train on (default: cuda)')

    parser.add_argument('--batch-size', default=16, type=int,
                        help='Batch size per GPU (default: 32)')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loading workers (default: 6)')

    parser.add_argument('--output-dir', required=True,
                        help='Path for saving features')
    parser.add_argument('--shard-id', default=0, type=int,
                        help='Shard id number. Must be between [0, num-shards)')
    parser.add_argument('--num-shards', default=1, type=int,
                        help='Number of shards to split the metadata-csv-filename')

    args = parser.parse_args()
    main(args)