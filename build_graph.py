import os
import sys
import glob
import argparse
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

import cv2
from sklearn.cluster import AgglomerativeClustering

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from models import ChangepointNetXent #, EdgeClassifier
from map_utils import mapPoint2Pixel
from structures import BehaviourGraph, Node

if sys.platform == "darwin":
    import matplotlib
    matplotlib.use('TkAgg')
else:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


class SampleChangepointDataset(Dataset):
    def __init__(
        self, 
        map_dir, 
        sampler, 
        # im_half_height=40, # floorplans
        # im_half_width=40 # floorplans
        im_half_height=60, # hand-drawn, satellite
        im_half_width=60 # hand-drawn, satellite
        ):
        self.half_height = im_half_height
        self.half_width = im_half_width
        self.height = self.half_height * 2 + 1
        self.width = self.half_width * 2 + 1

        print("Loading map")
        tmp = np.load(map_dir)
        self.map = torch.from_numpy(tmp['map']) / 255.0
        self.map = torch.swapaxes(torch.swapaxes(self.map, 0, 2), 1, 2)
        self.map_dims = self.map.shape
        self.bounds = tmp['bounds']
        self.res = tmp['res'].item()
        self.channels = self.map_dims[0]

        print("Generating data")
        self.data = sampler.sampleData()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample_pix = self.convertMap2Pixel(sample)
        crop = self.crop(sample_pix)
        return crop

    def getPixel(self, idx):
        sample = self.data[idx]
        return self.convertMap2Pixel(sample)

    def convertMap2Pixel(self, point):
        return mapPoint2Pixel(point, self.bounds, self.res)

    def crop(self, centre, padding=0.0):
        pos_ix, pos_iy = np.floor(centre).astype(int)

        xbl = pos_ix - self.half_width
        ybl = pos_iy - self.half_height
        xtr = pos_ix + self.half_width + 1
        ytr = pos_iy + self.half_height + 1

        map_xbl = max(xbl, 0)
        map_ybl = max(ybl, 0)
        map_xtr = min(xtr, self.map_dims[2])
        map_ytr = min(ytr, self.map_dims[1])

        im_xbl = map_xbl - xbl
        im_ybl = map_ybl - ybl
        im_xtr = self.width - (xtr - map_xtr)
        im_ytr = self.height - (ytr - map_ytr)

        im = torch.full((self.channels, self.height, self.width), fill_value=padding, dtype=torch.float)
        im[:, im_ybl:im_ytr, im_xbl:im_xtr] = self.map[:, map_ybl:map_ytr, map_xbl:map_xtr]

        return im

class ChangepointSamplerUniform:
    def __init__(self, bounds, x_res=0.25, y_res=0.25): # For hand-drawn and floorplans
    # def __init__(self, bounds, x_res=0.1, y_res=0.1): # For satellite map
        self.bounds =  bounds
        self.sample_x_res = x_res
        self.sample_y_res = y_res

        x_divs = (self.bounds[1][0] - self.bounds[0][0]) / self.sample_x_res
        y_divs = (-self.bounds[0][2] + self.bounds[1][2]) / self.sample_y_res
        print("*** ", x_divs, y_divs)
        self.x_dim = np.ceil(x_divs)
        self.y_dim = np.ceil(y_divs)

        # self.x_dim = np.floor((self.bounds[1][0] - self.bounds[0][0]) / self.sample_x_res) + 1
        # self.y_dim = np.floor((-self.bounds[0][2] + self.bounds[1][2]) / self.sample_y_res) + 1
        print("Dimensions: ", self.x_dim, self.y_dim)

        self.data = []

    def sampleData(self, eps=1e-8):
        self.data = []

        x = self.bounds[0][0]
        y = -self.bounds[1][2]
        num_y = 0
        while y < -self.bounds[0][2]:
            x = self.bounds[0][0]
            while x < self.bounds[1][0] and np.abs(x - self.bounds[1][0]) > eps:
                self.data.append(np.array([x, 0, y]))
                x += self.sample_x_res
            y += self.sample_y_res
            num_y += 1

        print(len(self.data))
        print(self.x_dim, self.y_dim, num_y)
        # sys.exit(0)

        return self.data

class GraphBuilder:
    def __init__(
        self,
        data_dir,
        map_file,
        changepoint_model_dir,
        edge_classifier_model_dir,
        batch_size=64
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.changepoint_model = ChangepointNetXent.load_from_checkpoint(changepoint_model_dir)
        self.edge_classifier_model = EdgeClassifier.load_from_checkpoint(edge_classifier_model_dir)

        self.map_dir = os.path.join(self.data_dir, map_file)
        tmp = np.load(self.map_dir)
        self.changepoint_sampler = ChangepointSamplerUniform(bounds=tmp['bounds'])
        self.changepoint_dataset = SampleChangepointDataset(self.map_dir, self.changepoint_sampler)

        self.changepoint_probs = None
        self.changepoint_threshold = 0.99

        self.edge_dataset = None


    def predictEdges(self, predicted_changepoints):
        self.edge_dataset = SampleEdgeDataset(
            self.map_dir, predicted_changepoints, im_half_height=60, im_half_width=60
        )
        loader = DataLoader(self.edge_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8, drop_last=False)
        trainer = pl.Trainer(accelerator='gpu', gpus=1)
        predictions = trainer.predict(self.edge_classifier_model, dataloaders=loader)
        probs = torch.cat(predictions, dim=0).numpy()
        predicted_edges = np.argmax(probs, axis=1)

        for datum, prediction in zip(self.edge_dataset.data, predicted_edges):
            if prediction != 3 :
                sx, sy = self.edge_dataset.convertMap2Pixel(datum[0:3])
                ex, ey = self.edge_dataset.convertMap2Pixel(datum[3:6])
                if prediction == 0:
                    c = 'b'
                elif prediction == 1:
                    c = 'k'
                elif prediction == 2:
                    c = 'r'
                else:
                    raise Exception("Invalid prediction")
                plt.arrow(sx, sy, ex - sx, ey - sy, color=c, length_includes_head=True, head_length=5, head_width=4)
                plt.savefig("/data/home/joel/datasets/source/bmapping/changepoint.png")

    def sampleChangepoints(self):
        loader = DataLoader(self.changepoint_dataset, batch_size=self.batch_size)
        trainer = pl.Trainer(accelerator='gpu', gpus=1)
        predictions = trainer.predict(self.changepoint_model, dataloaders=loader)
        self.changepoint_probs = torch.cat(predictions, dim=0).numpy()

    def postprocessChangepoints(self, points):
        X = points.copy()
        # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0).fit(X)
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5).fit(X) # Used for hand_com1_basement
        print(clustering.labels_)
        return clustering.labels_

    def predictChangepoints(self, postprocess=False):
        points = np.array(self.changepoint_dataset.data)
        thresholded = self.changepoint_probs[:, 0] > self.changepoint_threshold
        binary_im = thresholded.reshape(int(self.changepoint_sampler.y_dim), int(self.changepoint_sampler.x_dim))
        binary_im = binary_im.astype(np.uint8)
        plt.imshow(binary_im, origin='lower')
        plt.savefig("raw.png")

        kernelSize = (3, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        opening = cv2.morphologyEx(binary_im, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(binary_im, cv2.MORPH_CLOSE, kernel)
        binary_im = closing
        plt.imshow(binary_im, origin='lower')
        plt.savefig('raw2.png')

        ccount, cmask, cstats, _ = cv2.connectedComponentsWithStats(
            binary_im.astype(np.uint8), 4, cv2.CV_32S
        )

        prob_im = self.changepoint_probs[:, 0].reshape(int(self.changepoint_sampler.y_dim), int(self.changepoint_sampler.x_dim))
        predicted_changepoints = []

        for idx, comp in enumerate(cstats[1:]):
            decomp = self.decomposeComponent(cmask, prob_im, comp, idx+1)
            if decomp is not None:
                _, cluster_max_prob_samples = decomp

                for bx, by, bidx, bp in cluster_max_prob_samples:
                    predicted_changepoints.append(points[bidx])

        if postprocess:
            labels = self.postprocessChangepoints(predicted_changepoints)
            n_clusters = len(set(labels))
            clustered_changepoints = [[] for _ in range(n_clusters)]
            for point, label in zip(predicted_changepoints, labels):
                clustered_changepoints[label].append(point)
            merged_changepoints = [
                np.sum(cps, axis=0) / len(cps) 
                for cps in clustered_changepoints if len(cps) > 0
            ]

        changepoint_pix = np.array([self.changepoint_dataset.convertMap2Pixel(point) for point in predicted_changepoints])
        merged_pix = np.array([self.changepoint_dataset.convertMap2Pixel(point) for point in merged_changepoints])

        map_im = torch.swapaxes(torch.swapaxes(self.changepoint_dataset.map, 0, 2), 0, 1) * 255.0
        map_im = map_im.numpy().astype(np.uint8)

        plt.scatter(changepoint_pix[:, 0], changepoint_pix[:, 1], c='r', s=2.0, marker='o')
        plt.scatter(merged_pix[:, 0], merged_pix[:, 1], c='c', s=2.0, marker='o')
        # plt.imshow(self.changepoint_dataset.map, origin='lower')
        plt.imshow(map_im, origin='lower')
        # plt.savefig("/home/joel/research/behaviour_mapping/img.png")
        plt.savefig("changepoint_img.png")
        np.savez(
            'predicted_changepoints.npz',
            changepoints=merged_changepoints
        )

        # Save the changepoints as a graph (with no edges)
        nodes = [
            (self.changepoint_dataset.convertMap2Pixel(node), node, Node.CHANGEPOINT) 
            for node in predicted_changepoints
        ]

        tmp_graph = BehaviourGraph()
        tmp_graph.writeRawData2Graph(scene_dir="", nodes=nodes, edges=[], output_name="predicted_changepoints.json")

        return predicted_changepoints


    def decomposeComponent(self, mask, prob_im, component, idx):
        left, top, width, height, area = component
        if area <= 2:
            print("Filtering component ", idx)
            return None

        submask = (mask[top:top+height, left:left+width] == idx).astype(np.uint8)
        idx = 2
        convex_clusters = []
        for y in range(height):
            for x in range(width):
                if submask[y, x] == 1:
                    cluster = self.expandConvexCluster(submask, (x, y), idx)
                    if len(cluster) > 5:
                        convex_clusters.append((idx, cluster))
                    idx += 1

        prob_submask = prob_im[top:top+height, left:left+width]
        cluster_max_prob_samples = [self.findMaxProbSample(cluster, prob_submask) for _, cluster in convex_clusters]
        cluster_max_prob_samples = [
            (best_x + left, best_y + top, (best_x + left) + (best_y + top) * mask.shape[1], best_prob) 
            for best_x, best_y, best_prob in cluster_max_prob_samples
        ]

        return convex_clusters, cluster_max_prob_samples


    def findMaxProbSample(self, cluster, prob_mask):
        best_prob = -np.inf
        best_x = cluster[0][0]
        best_y = cluster[0][1]

        for x, y in cluster:
            if prob_mask[y, x] > best_prob:
                best_x = x
                best_y = y
                best_prob = prob_mask[y, x]

        return best_x, best_y, best_prob


    def expandConvexCluster(self, submask, coords, convex_idx):
        delta = 2.2
        scale = 1

        x, y = coords
        submask[y, x] = convex_idx
        nbs = self.getNeighbours4(submask, convex_idx, np.array([0, 0]), np.inf, x, y)
        cluster = [np.array([x, y]), *nbs]
        frontier = [*nbs]
        prev_cluster_size = np.inf

        if len(cluster) > 1:
            while len(cluster) != prev_cluster_size:
                prev_cluster_size = len(cluster)
                centre, (_, axlen) = self.getSmallestAxisLength(np.array(cluster))
                rmin = scale * axlen + delta
                next_frontier = []

                for x, y in frontier:
                    nbs = self.getNeighbours4(submask, convex_idx, centre, rmin, x, y)
                    next_frontier.extend(nbs)

                cluster.extend(next_frontier)
                frontier = next_frontier

        return cluster


    def getNeighbours4(self, submask, convex_idx, centre, rmin, x, y):
        h, w = submask.shape
        neighbours = []

        if x > 0 and submask[y, x-1] == 1 and np.linalg.norm(np.array([x-1, y]) - centre) < rmin:
            neighbours.append(np.array([x-1, y]))
            submask[y, x-1] = convex_idx
        if y > 0 and submask[y-1, x] == 1 and np.linalg.norm(np.array([x, y-1]) - centre) < rmin:
            neighbours.append(np.array([x, y-1]))
            submask[y-1, x] = convex_idx
        if x < w-1 and submask[y, x+1] == 1 and np.linalg.norm(np.array([x+1, y]) - centre) < rmin:
            neighbours.append(np.array([x+1, y]))
            submask[y, x+1] = convex_idx
        if y < h-1 and submask[y+1, x] == 1 and np.linalg.norm(np.array([x, y+1]) - centre) < rmin:
            neighbours.append(np.array([x, y+1]))
            submask[y+1, x] = convex_idx
        return neighbours


    def getSmallestAxisLength(self, points):
        mean = np.empty((0))
        mean, evecs, evals = cv2.PCACompute2(points.astype(np.float32), mean)
        return mean, evals.flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset", default="maps/satmap")
    parser.add_argument("--changepoint_model", type=str, help="Changepoint predictor model", default="")
    parser.add_argument("--edge_model", type=str, help="Edge predictor model", default="")
    args = parser.parse_args()

    gb = GraphBuilder(
        data_dir=args.dataset_dir,
        changepoint_model_dir=args.changepoint_model_path,
        edge_classifier_model_dir=args.edge_model_path,
    )
    gb.sampleChangepoints()
    preds = gb.predictChangepoints(postprocess=True)
    gb.predictEdges(preds)
