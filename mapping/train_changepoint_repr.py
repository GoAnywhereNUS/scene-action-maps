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

def collate_repr_data(batch):
    if len(batch) > 0:
        num_samples = batch[0][0].shape[0]
        ims = torch.cat([im for im, _ in batch])
        labels = torch.tensor([label for _, label in batch], dtype=torch.int)
        labels = labels.repeat_interleave(num_samples)
        return ims, labels
    else:
        return None


class ChangepointReprDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        sample_file="changepoint_samples.npz",
        # im_half_height=40, # floorplans
        # im_half_width=40, # floorplans
        im_half_height=60, # hand-drawn, satellite map
        im_half_width=60, # hand-drawn, satellite map
        num_views=2
    ):
        self.num_views = num_views
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
        views = [self.transforms(buffered_crop) for _ in range(self.num_views)]
        return torch.stack(views), self.labels[index]
        
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
        every_n_epochs=3
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        gpus=1, 
        max_epochs=args.max_epochs,
        default_root_dir=args.model_dir,
        logger=tb_logger,
        callbacks=[ckpt_callback]
    )

    # half_height/half_width == 40 for floorplans
    # half_height/half_width == 60 for hand-drawn
    # half_height/half_width == 60 for satellite map
    train_data = ChangepointReprDataset(
        args.dataset_dir, 
        sample_file="changepoint_samples.npz",
        num_views=args.num_views,
        im_half_height=args.half_im_size,
        im_half_width=args.half_im_size
        )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_repr_data)

    pl.seed_everything(args.random_seed)
    model = ChangepointNetRepr(
        temperature=args.loss_temp, 
        lr=args.lr,
        gamma=args.explr_gamma,
        misc_hparams=args
        )
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset",
        default="/data/home/joel/datasets/source/bmapping/maps/satmap")
    parser.add_argument("--model_dir", type=str, help="Path to save models to",
        default="/data/home/joel/datasets/models/cp_xent")
    parser.add_argument("--model_name", type=str, help="Name of model",
        default="cp_satmap_repr_v1")
    parser.add_argument("--log_dir", type=str, help="Directory for logging",
        default="/data/home/joel/datasets/logs")
    parser.add_argument("--lr", type=float, help="Learning rate",
        default=1e-2)
    parser.add_argument("--explr_gamma", type=float, help="Exponential LR scheduler gamma parameter",
        default=0.985)
    parser.add_argument("--batch_size", type=int, help="Training batch size",
        default=64)
    parser.add_argument("--max_epochs", type=int, help="Training max epochs",
        default=300)
    parser.add_argument("--num_views", type=int, help="Number of augmented views to generate per sample",
        default=2)
    parser.add_argument("--loss_temp", type=float, help="Temperature for supervised contrastive loss",
        default=0.1)
    parser.add_argument("--random_seed", type=int, help="Random seed",
        default=0)
    parser.add_argument("--half_im_size", type=int, help="Half size of image crop",
        default=60)
    args = parser.parse_args()

    train_model(args)

    # train_data = ChangepointReprDataset(
    #     "/Users/joel/Research/behaviour_mapping/bmapping/maps",
    #     sample_file="changepoint_samples.npz",
    #     im_half_height=40, 
    #     im_half_width=40
    # )
    # train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate_repr_data)

    # for i, batch in enumerate(train_loader):
    #     samples, labels = batch
    #     print(">>>>>", i)

    #     for j in range(4):
    #         print(labels[j])
    #         im = torch.swapaxes(torch.swapaxes(samples[j], 0, 2), 0, 1) * 255.0
    #         im = im.numpy().astype(np.uint8)
    #         plt.imshow(im, origin='lower')
    #         plt.scatter([41], [41], marker='x')
    #         plt.show()
