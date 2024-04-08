import os
import sys
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.distributions.uniform import Uniform
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import callbacks as pl_callbacks

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from map_utils import mapPoint2Pixel
from models import EdgeMatcherRepr


def collate_repr_data(batch):
    if len(batch) > 0:
        num_views = batch[0][0].shape[0]
        num_samples = len(batch)
        views = torch.cat([view for view, _, _, _ in batch], dim=0)
        labels = torch.cat([label for _, label, _, _ in batch], dim=0)
        map_idxs = torch.cat([midx for _, _, midx, _ in batch], dim=0)
        vert_idxs = torch.cat([vidx for _, _, _, vidx in batch], dim=0)

        # Construct positives mask. Positives are defined as:
        # 1. All augmented views of current sample
        # 2. a) If current sample has a labelled behaviour (behaviour >= 0),
        #       then include all out-edges from other vertices with same behaviour
        #    b) If current sample has no labelled behaviour (behaviour == -1),
        #       then include all other samples with no labelled behaviour as well,
        #       including samples from the current vertex
        # The if-condition for (2) is to differentiate between the same behaviour going
        # in different directions, when there is a valid behaviour or edge. If there
        # is no valid behaviour or edge, we do not try to make a distinction.
        mask_len = num_views * num_samples
        mask = torch.zeros(mask_len, mask_len).type(torch.bool)
    
        for srow in range(mask_len):
            for scol in range(mask_len):
                if (srow // num_views) == (scol // num_views):
                    # Same sample (and hence same label as well)
                    if srow != scol:
                        # Differs by augmentation (diagonal elements are self-similar)
                        # Equivalent to condition (1), i.e. all augmented views of
                        # current sample.
                        mask[srow, scol] = True
                else:
                    # Different samples
                    if labels[srow] == labels[scol]:
                        # Same label
                        if labels[srow] >= 0:
                            # Has a valid edge
                            if not (
                                map_idxs[srow] == map_idxs[scol]
                                and vert_idxs[srow] == vert_idxs[scol]
                                ):
                                # Must be samples from different vertices.
                                # Equivalent to condition (2a). Select all
                                # out-edges from other vertices that has the same
                                # (valid) label or behaviour.
                                mask[srow, scol] = True
                        # else:
                        #     # No valid edge. Equivalent to condition (2b).
                        #     # When there is no valid labelled behaviour, we
                        #     # use all samples with no labelled behaviour, including
                        #     # from the current vertex.
                        #     mask[srow, scol] = True

        mask = mask.float()
        return views, labels, mask, num_views
        # return views, labels, mask, num_views, map_idxs, vert_idxs

    else:
        return None

class EdgeReprDataset(Dataset):
    def __init__(
        self, 
        data_dir, 
        im_half_height=None,
        im_half_width=None,
        rotate_range=None,
        scale_range=None,
        num_views=2,
        balance_dataset=True
    ):
        assert num_views >= 2, "Too few views! (" + str(num_views) + " views)"
        self.num_views = num_views
        self.balance_dataset = balance_dataset

        print("Loading data")
        tmp = np.load(os.path.join(os.getcwd(), data_dir, 'preaug_edge_matching_samples.npz'))
        if tmp['halfh_halfw'].size != 0:
            raise AssertionError("Samples have fixed crop size. Not suited for data augmentation!")

        # Note: Assumes that the loaded dataset does not
        # have any 'filler' neighbours. I.e. all the neighbours
        # listed for each centre is a valid vertex in the map.
        # In this case, we further assume that the neighbour
        # count for all centres are the same, and equal to
        # min(max_num_neighbours, total_number_of_vertices_in_map)
        self.centres = tmp['centres'].astype(float)
        self.map_idxs = tmp['map_idxs'].astype(int)
        self.vertex_idxs = tmp['vertex_idxs'].astype(int)
        self.env_dirs = tmp['env_dirs']

        neighbour_counts = tmp['counts'].astype(int)
        assert len(neighbour_counts) > 0, "No data given!"        
        self.num_neighbours = neighbour_counts[0]

        self.neighbours = tmp['neighbours'].astype(float)
        self.neighbour_types = tmp['neighbour_types'].astype(int)
        self.neighbour_behaviours = tmp['neighbour_behaviours']

        print(dict(zip(*np.unique(self.neighbour_behaviours, return_counts=True))))

        self.half_height = im_half_height
        self.half_width = im_half_width
        # self.perms = None
        # self.perms_dim = tuple(tmp['perms_dim'])
        self.behaviour_flip_map = [2, 1, 0, 5, 4, 3]

        self.rotation_dist = (
            Uniform(-180, 180) if rotate_range is None
            else Uniform(rotate_range[0], rotate_range[1])
        )
        self.scale_dist = (
            Uniform(0.8, 1.0) if scale_range is None
            else Uniform(scale_range[0], scale_range[1])
        )

        self.height = self.half_height * 2 + 1
        self.width = self.half_width * 2 + 1
        self.buffered_half_height = self.half_height * 2
        self.buffered_half_width = self.half_width * 2

        print("Loading maps")
        self.maps = []
        for env_dir in self.env_dirs:
            tmp = np.load(os.path.join(os.getcwd(), data_dir, env_dir + '_map.npz'))
            env_map = torch.from_numpy(tmp['map']) / 255.0
            env_map = torch.swapaxes(torch.swapaxes(env_map, 0, 2), 1, 2)
            env_map_dims = env_map.shape
            env_bounds = tmp['bounds']
            env_res = tmp['res'].item()
            self.maps.append((env_map, env_map_dims, env_bounds, env_res))

        self.channels = self.maps[0][0].shape[0]

        if not self.balance_dataset:
            self.dataset_len = self.centres.shape[0] * self.num_neighbours
        else:
            # Process pre-augmentation data. Separate the nodes that fall
            # outside the map from the nodes inside the map (but which have
            # no associated behaviour)
            outside_neighbour_idxs = []
            inside_neighbour_idxs_null = []
            inside_neighbour_idxs_valid = []
            im_centre = torch.Tensor([self.half_width + 0.5, self.half_height + 0.5])

            for i in range(len(self.centres)):
                centre = self.centres[i]
                neighbours = self.neighbours[i]
                neighbour_behaviours = self.neighbour_behaviours[i]
                map_idx = self.map_idxs[i]
                _, _, _, res = self.maps[map_idx]
                rel_locs = torch.from_numpy(neighbours - centre)[:, [0, 2]]
                pix_locs = torch.floor(rel_locs / res + im_centre).int()

                for nidx, (loc, behaviour) in enumerate(zip(pix_locs, neighbour_behaviours)):
                    ix, iy = loc
                    inside = (0 <= ix and ix < self.width and 0 <= iy and iy < self.height)

                    if inside:
                        if behaviour == -1:
                            inside_neighbour_idxs_null.append((i, nidx))
                        else:
                            inside_neighbour_idxs_valid.append((i, nidx))
                    else:
                        outside_neighbour_idxs.append((i, nidx))

            # Sample some outside neighbours and inside (null behaviour) neighbours
            # (Hard-coded based on no. of behaviours)
            inside_nonnull_behaviour_count = len(inside_neighbour_idxs_valid)
            max_inside_sample_count = inside_nonnull_behaviour_count // 6
            max_outside_sample_count = inside_nonnull_behaviour_count // 6

            print("Inside valid: ", inside_nonnull_behaviour_count)
            print("Inside null: ", max_inside_sample_count, "/",  len(inside_neighbour_idxs_null))
            print("Outside: ", max_outside_sample_count, "/", len(outside_neighbour_idxs))

            if max_inside_sample_count < len(inside_neighbour_idxs_null):
                samples = torch.multinomial(
                    torch.Tensor([1 for _ in range(len(inside_neighbour_idxs_null))]), 
                    max_inside_sample_count, 
                    replacement=False
                )
                inside_neighbour_idxs_null = [inside_neighbour_idxs_null[s] for s in samples]
            
            if max_outside_sample_count < len(outside_neighbour_idxs):
                samples = torch.multinomial(
                    torch.Tensor([1 for _ in range(len(outside_neighbour_idxs))]),
                    max_outside_sample_count,
                    replacement=False
                )
                outside_neighbour_idxs = [outside_neighbour_idxs[s] for s in samples]

            neighbour_idxs = inside_neighbour_idxs_null + inside_neighbour_idxs_valid + \
                outside_neighbour_idxs
            tmp_centres = []
            tmp_neighbours = []
            tmp_neighbour_types = []
            tmp_neighbour_behaviours = []
            tmp_map_idxs = []
            tmp_vertex_idxs = []

            for cidx, nidx in neighbour_idxs:
                tmp_centres.append(self.centres[cidx])
                tmp_neighbours.append(self.neighbours[cidx][nidx])
                tmp_neighbour_types.append(self.neighbour_types[cidx][nidx])
                tmp_neighbour_behaviours.append(self.neighbour_behaviours[cidx][nidx])
                tmp_map_idxs.append(self.map_idxs[cidx])
                tmp_vertex_idxs.append(self.vertex_idxs[cidx])
            
            self.centres = tmp_centres
            self.neighbours = tmp_neighbours
            self.neighbour_types = tmp_neighbour_types
            self.neighbour_behaviours = tmp_neighbour_behaviours
            self.map_idxs = tmp_map_idxs
            self.vertex_idxs = tmp_vertex_idxs

            self.dataset_len = len(self.centres)

            print(">>>>>>> ", len(neighbour_idxs))
            # print(np.array(self.centres[:30]))
            # print(np.array(self.neighbours[:30]))
            # print(np.array(self.neighbour_behaviours[:30]))

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        if self.balance_dataset:
            centre = torch.from_numpy(self.centres[index])
            neighbour = torch.from_numpy(self.neighbours[index])
            neighbour_type = self.neighbour_types[index]
            behaviour = self.neighbour_behaviours[index]
            map_idx = self.map_idxs[index]
            vertex_idx = self.vertex_idxs[index]
        else:
            neighbour_idx = index % self.num_neighbours
            centre_idx = index // self.num_neighbours

            # Assume that there are no 'filler' neighbours
            centre = torch.from_numpy(self.centres[centre_idx])
            neighbour = torch.from_numpy(self.neighbours[centre_idx][neighbour_idx])
            neighbour_type = self.neighbour_types[centre_idx][neighbour_idx]
            behaviour = self.neighbour_behaviours[centre_idx][neighbour_idx]
            map_idx = self.map_idxs[centre_idx]
            vertex_idx = self.vertex_idxs[centre_idx]

        cpix = self.convertMap2Pixel(centre, map_idx)

        # Generate views by augmenting the data
        views = []
        labels = []
        p_flip, = torch.rand(1)
        flip_im = p_flip > 0.5
        if flip_im:
            behaviour = self.flipBehaviour(behaviour)

        for _ in range(self.num_views):
            # Augment the env map (up to rotation and scaling)
            cropped, corner = self.crop(cpix, map_idx, 
                self.buffered_half_height, self.buffered_half_width
            )
            transformed, T = self.rotateAndScale(cropped)
            env_map = transforms.functional.center_crop(
                transformed, (self.height, self.width)
            )

            # Rasterise the neighbour node
            neighbour_rel = (neighbour - centre)[[0, 2]].unsqueeze(dim=0).float()
            transformed = torch.matmul(T, neighbour_rel.T).flatten()
            im_centre = torch.Tensor([
                self.half_width + 0.5, self.half_height + 0.5
            ])
            _, _, _, res = self.maps[map_idx]
            transformed_pix = torch.floor(transformed / res + im_centre).int()
            neighbour_value = self.type2Val(neighbour_type)
            kp_im, is_neighbour_in_image = self.rasteriseKeypoint(
                transformed_pix, default_val=neighbour_value
            )

            # If neighbour is not rasterised inside image, kp_im
            # is basically empty, so we should reflect a null behaviour
            if not is_neighbour_in_image:
                behaviour = -1

            # Concatenate the map and keypoint image and flip if needed
            view = torch.cat([env_map, kp_im.unsqueeze(dim=0)], dim=0)
            if flip_im:
                view = transforms.functional.hflip(view)

            # Add the view
            views.append(view)
            labels.append(behaviour)

        labels = torch.tensor(labels, dtype=torch.int)
        current_map_idx = torch.full(labels.shape, map_idx)
        current_vert_idx = torch.full(labels.shape, vertex_idx)

        return torch.stack(views, dim=0), labels, current_map_idx, current_vert_idx

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

    def rasteriseKeypoint(self, point, default_val=0.0):
        # Rasterise the keypoint if it is within the crop.
        # Otherwise return an empty image.
        kp_im = torch.zeros((self.height, self.width))
        pos_ix, pos_iy = point
        inside = False

        if (0 <= pos_ix and pos_ix < self.width 
            and 0 <= pos_iy and pos_iy < self.height):
            kp_im[pos_iy, pos_ix] = default_val
            inside = True

        return kp_im, inside

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

    def flipBehaviour(self, behaviour):
        # This function is defined specifically for the behaviour 
        # set 'left', 'forward', 'right', where the permutations
        # have the structure: 
        # [d1_left, d1_forward, d1_right, d2_left, d2_forward, d2_right]
        # where d1 and d2 are different directions/orientations.
        return (
            self.behaviour_flip_map[behaviour]
            if behaviour >= 0 else behaviour
        )


def train_model(args):
    # Set torch seed for reproducibility
    pl.seed_everything(args.random_seed)

    print("=== Set up trainer")
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=args.log_dir,
        name='em',
        version=args.model_name
    )

    ckpt_callback = pl_callbacks.ModelCheckpoint(
        os.path.join(args.model_dir, args.model_name), 
        save_top_k=-1,
        every_n_epochs=3
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

    print("=== Load data")
    train_data = EdgeReprDataset('bmapping/maps/satmap',
        im_half_height=args.half_im_size, im_half_width=args.half_im_size, num_views=args.num_views
    )

    print("Length: ", len(train_data))
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, collate_fn=collate_repr_data
    )

    print("Length: ", len(train_loader))
    print("=== Instantiate model and train")
    model = EdgeMatcherRepr(
        temperature=args.loss_temp, 
        lr=args.lr, 
        gamma=args.explr_gamma,
        misc_hparams=args
    )
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, help="Path to save models to",
        default="/data/home/joel/datasets/models/em")
    parser.add_argument("--model_name", type=str, help="Name of model",
        # default="em_hand_repr_v5_resnet_balanced_nonnull")
        default="em_satmap_repr_v1_resnet_balanced_nonnull")
    parser.add_argument("--log_dir", type=str, help="Directory for logging",
        default="/data/home/joel/datasets/logs")
    parser.add_argument("--batch_size", type=int, help="Training batch size",
        default=64)
    parser.add_argument("--max_epochs", type=int, help="Training max epochs",
        default=200)
    parser.add_argument("--random_seed", type=int, help="Random seed",
        default=0)
    parser.add_argument("--num_workers", type=int, help="Number of workers",
        default=15)
    parser.add_argument("--loss_temp", type=float, help="Temperature for supervised contrastive loss",
        default=0.1)
    parser.add_argument("--num_views", type=int, help="Number of augmented views to generate per sample",
        default=2)
    parser.add_argument("--lr", type=float, help="Learning rate",
        default=1e-2)
    parser.add_argument("--explr_gamma", type=float, help="Exponential LR scheduler gamma parameter",
        default=0.98)
    parser.add_argument("--half_im_size", type=int, help="Half image crop size",
        default=100)
    args = parser.parse_args()

    train_model(args)

    # num_views = 2
    # train_data = EdgeReprDataset(
    #     'maps', im_half_height=100, im_half_width=100, num_views=num_views
    # )

    # train_loader = DataLoader(
    #     train_data, batch_size=20, shuffle=True, 
    #     num_workers=1, collate_fn=collate_repr_data
    # )

    # for batch in train_loader:
    #     _, labels, mask, _, map_idxs, vert_idxs = batch
    #     print(labels)
    #     map_idxs = list(map_idxs.numpy())
    #     vert_idxs = list(vert_idxs.numpy())
    #     labels = list(labels.numpy())
    #     print(list(zip(map_idxs, vert_idxs, labels)))
    #     print("=====")
    #     for row in mask:
    #         print(row)
    #     print("=====")
    #     break

    # # for i in range(20):
    # #     fig, axs = plt.subplots(1, num_views)
    # #     views, labels, _ = train_data[i]
        
    # #     for idx, view in enumerate(views):
    # #         map_im = view[:3]
    # #         map_im = torch.swapaxes(torch.swapaxes(map_im, 0, 2), 0, 1)
    # #         axs[idx].imshow(map_im, origin='lower')

    # #         kp_im = view[-1]
    # #         y_idxs, x_idxs = np.where(kp_im.numpy() > 0.)
    # #         assert len(y_idxs) <= 1 and len(x_idxs) <=1, "Too many keypoints in image!"

    # #         if len(y_idxs) == 1 and len(x_idxs) == 1:
    # #             x_idx, y_idx = x_idxs[0], y_idxs[0]
    # #             if kp_im[y_idx, x_idx] < 1.:
    # #                 axs[idx].scatter(x_idx, y_idx, marker='o', c='#fa8072')
    # #             else:
    # #                 axs[idx].scatter(x_idx, y_idx, marker='o', c='g')

    # #             if labels[idx] % 3 == 0:
    # #                 c = 'b'
    # #             elif labels[idx] % 3 == 1:
    # #                 c = 'k'
    # #             else:
    # #                 c = 'r'

    # #             if labels[idx] >= 0:
    # #                 axs[idx].arrow(100, 100, x_idx - 100, y_idx - 100, 
    # #                     color=c, linestyle='-', overhang=0.5,
    # #                     length_includes_head=True, head_length=5, head_width=4)

    # #         axs[idx].scatter(100, 100, marker='x')

    # #     plt.show()