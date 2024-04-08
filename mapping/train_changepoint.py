import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from map_utils import mapPoint2Pixel
from models import ChangepointNetXent, ChangepointNetRepr

class ChangepointDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        sample_file="changepoint_samples.npz", 
        # im_half_height=40, # floorplans
        # im_half_width=40 # floorplans
        im_half_height=60, # hand-drawn, satellite map
        im_half_width=60 # hand-drawn, satellite map
    ):
        self.half_height = im_half_height
        self.half_width = im_half_width
        self.height = self.half_height * 2 + 1
        self.width = self.half_width * 2 + 1

        self.buffered_half_height = self.half_height * 2
        self.buffered_half_width = self.half_width * 2
        self.buffered_height = self.buffered_half_height * 2 + 1
        self.buffered_width = self.buffered_half_width * 2 + 1

        self.transforms = transforms.Compose([
            transforms.RandomAffine(
                degrees=(-180, 179), 
                scale=(0.8, 1.0), 
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.CenterCrop(size=(self.height, self.width))
        ])

        print("Preparing data")
        tmp = np.load(os.path.join(data_dir, sample_file))
        pos_samples = tmp['pos_samples']
        neg_samples = tmp['neg_samples']
        self.data = np.vstack((pos_samples, neg_samples))
        self.labels = (
            [0 for i in range(len(pos_samples))] +
            [1 for i in range(len(neg_samples))]
        )

        print("Loading maps")
        self.maps = []
        for env_dir in tmp['dirs']:
            map_files = np.load(os.path.join(data_dir, env_dir + '_map.npz'))
            map_tensor = torch.from_numpy(map_files['map']) / 255.0
            map_tensor = torch.swapaxes(torch.swapaxes(map_tensor, 0, 2), 1, 2)
            map_dims = map_tensor.shape
            map_bounds = map_files['bounds']
            map_res = map_files['res'].item()
            self.maps.append((map_tensor, map_dims, map_bounds, map_res))

        self.channels = self.maps[0][0].shape[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        point = self.data[index]
        pix = self.convertMap2Pixel(point)
        buffered_crop = self.crop(pix, int(point[-1]))
        transformed_crop = self.transforms(buffered_crop)
        return (transformed_crop, self.labels[index])
        
    def convertMap2Pixel(self, point):
        coords = point[:3]
        map_idx = int(point[-1])
        _, _, bounds, res = self.maps[map_idx]
        return mapPoint2Pixel(coords, bounds, res)

    def crop(self, centre, map_idx, padding=0.0):
        map, map_dims, _, _ = self.maps[map_idx]
        pos_ix, pos_iy = np.floor(centre).astype(int)

        xbl = pos_ix - self.buffered_half_width
        ybl = pos_iy - self.buffered_half_height
        xtr = pos_ix + self.buffered_half_width + 1
        ytr = pos_iy + self.buffered_half_height + 1

        map_xbl = max(xbl, 0)
        map_ybl = max(ybl, 0)
        map_xtr = min(xtr, map_dims[2])
        map_ytr = min(ytr, map_dims[1])

        im_xbl = map_xbl - xbl
        im_ybl = map_ybl - ybl
        im_xtr = self.buffered_width - (xtr - map_xtr)
        im_ytr = self.buffered_height - (ytr - map_ytr)

        im = torch.full(
            (self.channels, self.buffered_height, self.buffered_width), 
            fill_value=padding, dtype=torch.float
            )
        im[:, im_ybl:im_ytr, im_xbl:im_xtr] = map[:, map_ybl:map_ytr, map_xbl:map_xtr]

        return im

def train_model(args):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name='cp',
        version=args.model_name 
    )

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, args.model_name), 
        save_top_k=-1,
        every_n_epochs=1
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        gpus=2,
        strategy="ddp",
        max_epochs=args.max_epochs,
        default_root_dir=args.model_dir,
        logger=tb_logger,
        callbacks=[ckpt_callback]
    )

    train_data = ChangepointDataset(args.dataset_dir, sample_file="changepoint_samples.npz")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    pl.seed_everything(args.random_seed)
    model = ChangepointNetXent()
    encoder_model = ChangepointNetRepr.load_from_checkpoint(
        # '/data/home/joel/datasets/models/cp_xent/cp_repr_decayed_v1/epoch=98-step=27621.ckpt'
        # '/data/home/joel/datasets/models/cp_xent/cp_hand_repr_v1/epoch=299-step=104100.ckpt'
        '/data/home/joel/datasets/models/cp_xent/cp_satmap_repr_v1/epoch=296-step=185031.ckpt'
    )
    model.encoder = encoder_model.encoder
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
        default="/data/home/joel/datasets/source/bmapping/maps/satmap")
    parser.add_argument("--model_dir", type=str, help="Path to save models to",
        default="/data/home/joel/datasets/models/cp_xent")
    parser.add_argument("--model_name", type=str, help="Name of model",
        # default="cp_hand_ft_repr_v1_corrected")
        default="cp_satmap_ft_repr_v1")
    parser.add_argument("--log_dir", type=str, help="Directory for logging",
        default="/data/home/joel/datasets/logs")
    parser.add_argument("--batch_size", type=int, help="Training batch size",
        default=64)
    parser.add_argument("--max_epochs", type=int, help="Training max epochs",
        default=40)
    parser.add_argument("--random_seed", type=int, help="Random seed",
        default=0)
    parser.add_argument('--num_workers', type=int, help="Number of workers for dataloader",
        default=16)
    args = parser.parse_args()

    train_model(args)

    # train_data = ChangepointDataset(
    #     "/Users/joel/Research/behaviour_mapping/bmapping/maps",
    #     sample_file="changepoint_samples.npz",
    #     im_half_height=40, 
    #     im_half_width=40
    # )
    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # for i, batch in enumerate(train_loader):
    #     samples, labels = batch
    #     print(labels[0])
    #     im = torch.swapaxes(torch.swapaxes(samples[0], 0, 2), 0, 1) * 255.0
    #     im = im.numpy().astype(np.uint8)
    #     plt.imshow(im, origin='lower')
    #     # plt.imshow(samples[0, :, :].numpy().squeeze(), origin="lower")
    #     plt.scatter([41], [41], marker='x')
    #     # plt.scatter([61], [61], marker='x')
    #     plt.show()