import os
import sys
import cv2
import argparse
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.uniform import Uniform
from torchvision import transforms

from map_utils import mapPoint2Pixel

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from models import EdgeMatcher, EdgeMatcherRepr

class EdgeDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        augment=False, 
        rescale_map=None,
        im_half_height=None,
        im_half_width=None,
        n_max_neighbours=None,
        rotate_range=None,
        scale_range=None
    ):
        self.augment = augment

        print("Loading data")
        tmp = np.load(os.path.join(os.getcwd(), data_dir, 'preaug_edge_matching_samples.npz'))

        print(tmp['halfh_halfw'])
        if tmp['halfh_halfw'].size != 0:
            assert self.augment, "Samples have fixed crop size. Not suited for data augmentation!"

        self.centres = tmp['centres'].astype(float)
        self.neighbours = tmp['neighbours'].astype(float)
        self.neighbour_types = tmp['neighbour_types'].astype(int)
        self.neighbour_counts = tmp['counts'].astype(int)
        self.map_idxs = tmp['map_idxs'].astype(int)
        self.env_dirs = tmp['env_dirs']

        if self.augment:
            self.half_height = im_half_height
            self.half_width = im_half_width
            self.perms = None
            self.perms_dim = tuple(tmp['perms_dim'])
            self.neighbour_behaviours = tmp['neighbour_behaviours']
            self.n_max_neighbours = n_max_neighbours

            self.rotation_dist = (
                Uniform(-180, 180) if rotate_range is None
                else Uniform(rotate_range[0], rotate_range[1])
            )
            self.scale_dist = (
                Uniform(0.8, 1.0) if scale_range is None
                else Uniform(scale_range[0], scale_range[1])
            )
        else:
            self.half_height, self.half_width = tmp['halfh_halfw']
            self.perms = torch.from_numpy(tmp['perms']).float()
            self.perms_dim = None
            self.neighbour_behaviours = None
            self.n_max_neighbours = None

            self.rotation_dist = None
            self.scale_dist = None

        self.height = self.half_height * 2 + 1
        self.width = self.half_width * 2 + 1
        self.buffered_half_height = self.half_height * 2
        self.buffered_half_width = self.half_width * 2

        print("Loading maps")
        self.maps = []
        for env_dir in self.env_dirs:
            tmp = np.load(os.path.join(os.getcwd(), data_dir, env_dir + '_map.npz'))
            np_map = tmp['map']
            if rescale_map:
                np_map = cv2.resize(
                    np_map, dsize=None, fx=rescale_map, fy=rescale_map, interpolation=cv2.INTER_CUBIC
                )

            env_map = torch.from_numpy(np_map) / 255.0
            env_map = torch.swapaxes(torch.swapaxes(env_map, 0, 2), 1, 2)
            env_map_dims = env_map.shape
            env_bounds = tmp['bounds']
            env_res = tmp['res'].item()
            if rescale_map:
                env_res = (env_bounds[1][0] - env_bounds[0][0]) / np_map.shape[1]
                print('---')
                print(env_res, env_bounds, env_map_dims)

            self.maps.append((env_map, env_map_dims, env_bounds, env_res))

            np_map = torch.swapaxes(torch.swapaxes(env_map, 0, 2), 0, 1)
            # plt.imshow(np_map, origin='lower')

        self.channels = self.maps[0][0].shape[0]

    def __len__(self):
        return self.centres.shape[0]

    def __getitem__(self, index):
        count = self.neighbour_counts[index]
        centre = self.centres[index]
        neighbours = self.neighbours[index][:count, :]
        neighbour_types = self.neighbour_types[index][:count]
        behaviours = self.neighbour_behaviours[index][:count]
        map_idx = self.map_idxs[index]

        # Get crop of map and augment if needed
        cpix = self.convertMap2Pixel(centre, map_idx)
        flipped = False

        if self.augment:
            cropped, corner = self.crop(cpix, map_idx, 
                self.buffered_half_height, self.buffered_half_width)
            transformed, T = self.rotateAndScale(cropped)
            env_map = transforms.functional.center_crop(
                transformed, (self.height, self.width))

            rel_kps = torch.from_numpy(neighbours - centre).float()[:, [0,2]]
            transformed_rel_kps = torch.matmul(T, rel_kps.T).T
            im_centre = torch.Tensor([self.half_width + 0.5, self.half_height + 0.5])
            _, _, _, res = self.maps[map_idx]
            transformed_kp_pix = torch.floor(
                (transformed_rel_kps / res) + im_centre).int()
            
            kp_ims = []
            perms = torch.zeros(self.perms_dim)
            for idx, (kp, ntype) in enumerate(zip(transformed_kp_pix, neighbour_types)):
                kp_im = self.rasteriseKeypointInsideCrop(
                    kp, default_val=self.type2Val(ntype)
                )

                if kp_im is not None:
                    if behaviours[idx] > -1:
                        perms[len(kp_ims), behaviours[idx]] = 1.
                    kp_ims.append(kp_im)

            kp_ims += [torch.zeros((self.height, self.width))
                for _ in range(self.n_max_neighbours - len(kp_ims))]
            kp_ims = torch.stack(kp_ims)

            env_map, kp_ims, flipped = self.flip(env_map, kp_ims)

        else:
            env_map, corner = self.crop(cpix, map_idx, self.half_h, self.half_w)
            perms = self.perms[index]
            kp_ims = torch.stack([
                (self.rasteriseKeypoint(
                    self.convertMap2Pixel(neighbour, map_idx), 
                    corner, 
                    default_val=self.type2Val(neighbour_type)
                    ) 
                    if idx < count else self.rasteriseKeypoint(cpix, corner))
                for idx, (neighbour, neighbour_type) in enumerate(zip(neighbours, neighbour_types))
            ])

        shuffle_idxs = torch.randperm(self.perms_dim[0])
        kp_ims = kp_ims[shuffle_idxs, :, :]
        perms = self.flipPerms(perms[shuffle_idxs, :]) if flipped else perms[shuffle_idxs, :]

        return env_map, kp_ims, perms

    def type2Val(self, neighbour_type):
        if neighbour_type == 0:
            return 1.0
        elif neighbour_type == 1:
            return 0.5
        else:
            raise NotImplementedError

    def convertMap2Pixel(self, point, map_idx):
        _, _, bounds, res = self.maps[map_idx]
        return mapPoint2Pixel(point, bounds, res)

    def rasteriseKeypoint(self, point, corner, default_val=0.0):
        kp_im = torch.zeros(self.height, self.width)
        xbl, ybl = corner
        pos_ix, pos_iy = np.floor(point).astype(int)
        pos_ix, pos_iy = self.clamp(pos_ix - xbl, pos_iy - ybl)
        kp_im[pos_iy, pos_ix] = default_val
        return kp_im

    def rasteriseKeypointInsideCrop(self, point, default_val=0.0):
        # Rasterise the keypoint, then check if it is inside the crop.
        # If not, return None.
        pos_ix, pos_iy = point
        if (0 <= pos_ix and pos_ix < self.width 
            and 0 <= pos_iy and pos_iy < self.height):
            kp_im = torch.zeros((self.height, self.width))
            kp_im[pos_iy, pos_ix] = default_val
            return kp_im

        return None

    def clamp(self, px, py):
        return (
            min(self.width-1, max(0, px)),
            min(self.height-1, max(0, py))
        )

    def crop(self, centre, map_idx, half_h, half_w, padding=1.0):
        env_map, env_map_dims, _, _ = self.maps[map_idx]
        pos_ix, pos_iy = np.floor(centre).astype(int)
        height = half_h * 2 + 1
        width = half_w * 2 + 1

        xbl = pos_ix - half_w
        ybl = pos_iy - half_h
        xtr = pos_ix + half_w + 1
        ytr = pos_iy + half_h + 1

        map_xbl = max(xbl, 0)
        map_ybl = max(ybl, 0)
        map_xtr = min(xtr, env_map_dims[2])
        map_ytr = min(ytr, env_map_dims[1])

        im_xbl = map_xbl - xbl
        im_ybl = map_ybl - ybl
        im_xtr = width - (xtr - map_xtr)
        im_ytr = height - (ytr - map_ytr)

        im = torch.full((self.channels, height, width), fill_value=padding, dtype=torch.float)
        im[:, im_ybl:im_ytr, im_xbl:im_xtr] = env_map[:, map_ybl:map_ytr, map_xbl:map_xtr]

        return im, (xbl, ybl)

    def rotateAndScale(self, im):
        # Sample a rotation and scale
        angle = float(self.rotation_dist.sample())
        scale = float(self.scale_dist.sample())

        # Transform the image
        transformed = transforms.functional.affine(
            im, angle=angle, translate=[0, 0], scale=scale, shear=0.,
            interpolation=transforms.functional.InterpolationMode.NEAREST)
        
        # Compute the transform
        angle = angle * np.pi / 180.
        R = torch.Tensor([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        S = torch.eye(2) * scale
        T = torch.matmul(R, S).type(torch.float)

        return transformed, T

    def flip(self, im, kp_ims):
        p = torch.rand(1)[0]
        if p > 0.5:
            im = transforms.functional.hflip(im)
            kp_ims = transforms.functional.hflip(kp_ims)
            return im, kp_ims, True
        return im, kp_ims, False

    def flipPerms(self, perms):
        # CAUTION: This implementation of flipPerms is
        # defined specifically for the behaviour set
        # 'left', 'forward', 'right', where the permutations
        # have the structure: 
        # [d1_left, d1_forward, d1_right, d2_left, d2_forward, d2_right]
        # where d1 and d2 are different directions/orientations.
        perms = perms[:, [2, 1, 0, 5, 4, 3]]
        return perms


def train_model(args):
    print("=== Set up trainer")
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name='em',
        version=args.model_name
    )

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, args.model_name), 
        save_top_k=-1,
        every_n_epochs=2
    )

    trainer = pl.Trainer(
        accelerator='gpu', 
        gpus=2, 
        strategy='ddp',
        max_epochs=args.max_epochs,
        default_root_dir=args.model_dir,
        logger=tb_logger,
        callbacks=[ckpt_callback]
    )

    print("=== Load data")
    train_data = EdgeDataset('bmapping/maps/hand_drawn',
        augment=True, im_half_height=args.half_im_size, im_half_width=args.half_im_size,
        rescale_map=args.map_scale_factor, n_max_neighbours=10)
    print("Length: ", len(train_data))
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    print("Length: ", len(train_loader))

    print("=== Instantiate model and train")
    pl.seed_everything(args.random_seed)
    model = EdgeMatcher(misc_hparams=args)

    if args.pretrained_feat_encoder:
        print(">>> Loading pretrained representation model")
        repr_model = EdgeMatcherRepr.load_from_checkpoint(
            args.pretrained_feat_encoder
        )
        model.adapter = repr_model.adapter
        model.backbone = repr_model.encoder

    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to save models to",
        default="/data/home/joel/datasets/models/em")
    parser.add_argument("--model_name", type=str, help="Name of model",
        # default="em_floor_ft_repr_v5_resnet_balanced_nonnull")
        # default="em_hand_v6_resnet_negloss")
        default="em_hand_ft_repr_v5_resnet_negloss")
    parser.add_argument("--log_dir", type=str, help="Directory for logging",
        default="/data/home/joel/datasets/logs")
    parser.add_argument("--batch_size", type=int, help="Training batch size",
        default=32)
    parser.add_argument("--max_epochs", type=int, help="Training max epochs",
        default=200)
    parser.add_argument("--random_seed", type=int, help="Random seed",
        default=0)
    parser.add_argument("--num_workers", type=int, help="Number of workers",
        default=16)
    parser.add_argument("--pretrained_feat_encoder", type=str, help="Path to pretrained encoder",
        # default="/data/home/joel/datasets/models/em/em_hand_repr_v2_full_nonnull/epoch=197-step=118206.ckpt")
        # default="/data/home/joel/datasets/models/em/em_repr_v1/epoch=158-step=111300.ckpt")
        # default="/data/home/joel/datasets/models/em/em_hand_repr_v1/epoch=197-step=17820.ckpt")
        # default="/data/home/joel/datasets/models/em/em_hand_repr_v3_balanced_null/epoch=197-step=37818.ckpt")
        default="/data/home/joel/datasets/models/em/em_hand_repr_v5_resnet_balanced_nonnull/epoch=197-step=56628.ckpt")
        # default="/data/home/joel/datasets/models/em/em_floor_repr_v3_balanced_nonnull/epoch=197-step=43560.ckpt")
        # default="/data/home/joel/datasets/models/em/em_satmap_repr_v1_resnet_balanced_nonnull/epoch=197-step=97416.ckpt")
        # default="/data/home/joel/datasets/models/em/em_floor_repr_v5_resnet_balanced_nonnull/epoch=197-step=43560.ckpt")
        # default="")
    parser.add_argument("--half_im_size", type=int, help="Half size of image crop",
        default=100)
    parser.add_argument("--map_scale_factor", type=float, help="Factor by which to rescale map",
        default=None)
        # default=0.6)
    parser.add_argument("--neg_scale_factor", type=float, help="Factor by which to scale non-match loss",
        default=5.0)
    args = parser.parse_args()

    train_model(args)

    # # Test data loading
    # # train_data = EdgeDataset('maps')
    # train_data = EdgeDataset('maps', 
    #     augment=True, im_half_height=100, im_half_width=100,
    #     n_max_neighbours=10)

    # for i in range(20):
    #     crop, kp_ims, perms = train_data[i]
    #     im = torch.zeros((3, 201, 201))
    #     im[:, :, :] = crop
    #     im = torch.swapaxes(torch.swapaxes(im, 0, 2), 0, 1)

    #     cx, cy = (100, 100)

    #     verts = []
    #     for kp_im in kp_ims:
    #         y_idxs, x_idxs = np.where(kp_im.numpy() > 0.)
    #         print(">>> ", y_idxs, x_idxs)
    #         assert len(y_idxs) <= 1 and len(x_idxs) <=1, "Too many keypoints in image!"

    #         if len(y_idxs) == 1 and len(x_idxs) == 1:
    #             x_idx, y_idx = x_idxs[0], y_idxs[0]
    #             if kp_im[y_idx, x_idx] < 1.:
    #                 plt.scatter(x_idx, y_idx, marker='o', c='#fa8072')
    #             else:
    #                 plt.scatter(x_idx, y_idx, marker='o', c='g')
    #             verts.append((x_idx, y_idx))
    #         else:
    #             verts.append(None)

    #     for idx, out_edges in enumerate(perms):
    #         idxs, = np.where(out_edges > 0)
    #         if len(idxs) == 1:
    #             edge = idxs[0]
    #             if edge % 3 == 0:
    #                 c = 'b'
    #             elif edge % 3 == 1:
    #                 c = 'k'
    #             elif edge % 3 == 2:
    #                 c = 'r'
    #             else:
    #                 raise NotImplementedError

    #             if edge // 3 == 0:
    #                 ls = ':'
    #             elif edge // 3 == 1:
    #                 ls = '-'
    #             else:
    #                 raise NotImplementedError

    #             vx, vy = verts[idx]
    #             plt.arrow(cx, cy, vx - cx, vy - cy, color=c, linestyle=ls,
    #                 head_width=2.5, head_length=4, length_includes_head=True)
            
    #     plt.scatter(cx, cy, marker='x', c='r')
    #     plt.imshow(im, origin='lower')
    #     plt.show()

    # # Test the model
    # test_data = EdgeDataset("maps")
    # test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

    # # model = EdgeMatcher.load_from_checkpoint(
    # #     "/home/j/joell/datasets/source/saved_models/edge_matcher_xent_all.ckpt"
    # # )
    # model = EdgeMatcher.load_from_checkpoint(
    #     "/data/home/joel/datasets/models/em/logs/lightning_logs/version_0/checkpoints/epoch=59-step=20160.ckpt"
    # )

    # tester = pl.Trainer(accelerator='gpu', gpus=1)
    # tester.test(model, dataloaders=test_loader)
