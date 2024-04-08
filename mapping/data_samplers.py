import os
import numpy as np
import argparse
from operator import itemgetter

from structures import BehaviourGraph, Intention
from map_utils import mapPoint2Pixel


class ChangepointSampler:
    def __init__(
        self, 
        data_dir, 
        env_dirs, 
        rng, 
        angular_change_deg=20,
        changepoint_radius=0.5
    ):
        self.data_dir = data_dir
        self.env_dirs = env_dirs
        self.rng = rng

        self.max_edge_angular_change_rad = np.cos(angular_change_deg * np.pi / 180.0)
        self.changepoint_radius = changepoint_radius

    def setupMap(self, env_dir):
        path = os.path.join(os.getcwd(), self.data_dir)

        print("Loading map from ", path)
        tmp = np.load(os.path.join(path, env_dir + '_map.npz'))
        self.map = tmp['map']
        self.map_dims = self.map.shape
        self.bounds = tmp['bounds']
        self.res = tmp['res']

        graph = BehaviourGraph()
        graph.loadGraph(os.path.join(path, env_dir + '_graph.json'))
        graph.initialise()

        changepoints = [(pix, pos) for pix, pos, label in graph.graph_nodes if not label]
        self.changepoint_coords = np.array([[pos[0], pos[2]] for _, pos in changepoints])

        out_vert_idxs = [vert['out_edges'] for vert in graph.vertices if not vert['place_node']]
        self.out_verts = [
            [graph.graph_nodes[idx][1] for idx, _, _ in out_verts] 
            for out_verts in out_vert_idxs
        ]
        self.out_verts = [np.array([[px, py] for px, _, py in points]) for points in self.out_verts]

        self.max_y = -self.bounds[0][2]
        self.min_y = -self.bounds[1][2]
        self.max_x = self.bounds[1][0]
        self.min_x = self.bounds[0][0]

        self.x_ext = self.max_x - self.min_x
        self.y_ext = self.max_y - self.min_y       
        

    def checkValidSample(self, sample, changepoint_idx):
        changepoint = self.changepoint_coords[changepoint_idx]
        out_verts = self.out_verts[changepoint_idx] 

        # Compare outgoing edge vectors wrt changepoint
        # against outgoing edge vectors wrt sampled point,
        # and make sure their angles don't deviate too much
        changepoint_edges = out_verts - changepoint
        changepoint_edges_norm = changepoint_edges / np.linalg.norm(changepoint_edges, axis=1)[:, None]
        sample_edges = out_verts - sample
        sample_edges_norm = sample_edges / np.linalg.norm(sample_edges, axis=1)[:, None]
        pairwise_dotprod = np.abs(np.sum(changepoint_edges_norm * sample_edges_norm, axis=1))

        dist = np.linalg.norm(sample - changepoint)
        return (
            dist < self.changepoint_radius and
            np.all(pairwise_dotprod > self.max_edge_angular_change_rad)
        )

    def samplePositive(self, map_idx, changepoint_idx):
        # Sample a point in the vicinity of the changepoint
        centre_x, centre_y = self.changepoint_coords[changepoint_idx]
        r = np.sqrt(self.rng.random()) * self.changepoint_radius
        theta = self.rng.random() * 2 * np.pi
        x = centre_x + r * np.cos(theta)
        y = centre_y + r * np.sin(theta)

        sample = np.array([x, y])

        if self.checkValidSample(sample, changepoint_idx):
            point = np.array([x, 0, y])
            return np.hstack((point, np.array([map_idx])))

        return None

    def sampleNegative(self, map_idx):
        x = (self.rng.random() * self.x_ext) + self.min_x
        y = (self.rng.random() * self.y_ext) + self.min_y

        sample = np.array([x, y])
        dists = np.linalg.norm(self.changepoint_coords - sample, axis=1)
        near_changepoint = dists < self.changepoint_radius
        
        if np.any(near_changepoint):
            idxs, = np.where(near_changepoint)
            for idx in idxs:
                if self.checkValidSample(sample, idx):
                    return None
                
        point = np.array([x, 0, y])
        return np.hstack((point, np.array([map_idx])))


    def generateData(
        self, 
        num_pos_samples_per_changepoint=20,
        neg_to_pos_sample_ratio=1.0
        ):
        pos_samples = []
        neg_samples = []

        for map_idx, env_dir in enumerate(self.env_dirs):
            self.setupMap(env_dir)

            print("Sampling positive")
            for changepoint_idx in range(len(self.changepoint_coords)):
                count = 0
                while count < num_pos_samples_per_changepoint:
                    sample = self.samplePositive(map_idx, changepoint_idx)
                    if sample is not None:
                        pos_samples.append(sample)
                        count += 1

            print("Sampling negative")
            num_required_neg_samples = int(len(pos_samples) * neg_to_pos_sample_ratio)
            while len(neg_samples) < num_required_neg_samples:
                sample = self.sampleNegative(map_idx)
                if sample is not None:
                    neg_samples.append(sample)

        np.savez(
            os.path.join(os.getcwd(), self.data_dir, 'changepoint_samples.npz'),
            dirs=np.array(self.env_dirs),
            pos_samples=pos_samples,
            neg_samples=neg_samples
        )


class ReducedEdgeMatchingSampler:
    def __init__(
        self,
        data_dir,
        env_dirs,
        rng,
        vertex_radius=0.5,
        n_max_neighbours=10,
        n_directions=2,
        n_behaviours=None,
        max_edge_angular_change_deg=15.
    ):
        self.data_dir = data_dir
        self.env_dirs = env_dirs
        self.rng = rng
        self.vertex_radius = vertex_radius
        self.n_max_neighbours = n_max_neighbours
        self.n_directions = n_directions
        self.n_behaviours = len(Intention) if n_behaviours is None else n_behaviours
        self.max_edge_angular_change_rad = np.cos(max_edge_angular_change_deg * np.pi / 180.)

    def setupMap(self, env_dir):
        path = os.path.join(os.getcwd(), self.data_dir)

        print("Loading map")
        tmp = np.load(os.path.join(path, env_dir + "_map.npz"))
        self.map = tmp['map']
        self.map_dims = self.map.shape
        self.bounds = tmp['bounds']
        self.res = tmp['res']

        print("Loading graph")
        graph = BehaviourGraph()
        graph.loadGraph(os.path.join(path, env_dir + "_graph.json"))
        graph.initialise()

        # Manually get out edges because the behaviour graph
        # structure does not specify edge_dir in its vertex out_edges
        self.out_edges = [[] for i in range(len(graph.graph_nodes))]
        for edge_src, edge_dst, edge_int, edge_dir in graph.graph_edges:
            self.out_edges[edge_src].append((edge_dst, edge_int, edge_dir))

        out_vert_idxs = [vert['out_edges'] for vert in graph.vertices]
        self.out_verts = [
            [graph.graph_nodes[idx][1] for idx, _, _ in out_verts] 
            for out_verts in out_vert_idxs
        ]
        self.out_verts = [np.array([[px, py] for px, _, py in points]) for points in self.out_verts]

        positions = np.array([pos for _, pos, _ in graph.graph_nodes])
        self.vertex_coords_2d = positions[:, [0, 2]]
        self.vertex_types = [label for _, _, label in graph.graph_nodes]
        self.pairwise_dists = np.linalg.norm(
            self.vertex_coords_2d - self.vertex_coords_2d[:, None], 
            axis=2
        )

    def convertMap2Pixel(self, point):
        return mapPoint2Pixel(point, self.bounds, self.res)

    def samplePoint(self, vertex_idx):
        vertex_x, vertex_y = self.vertex_coords_2d[vertex_idx]
        r = np.sqrt(self.rng.random()) * self.vertex_radius
        theta = self.rng.random() * 2 * np.pi
        x = vertex_x + r * np.cos(theta)
        y = vertex_y + r * np.sin(theta)

        if not self.checkValidSample(np.array([x, y]), vertex_idx):
            return None

        return np.array([x, 0, y])

    def checkValidSample(self, sampled_vertex, vertex_idx):
        orig_vertex = self.vertex_coords_2d[vertex_idx]
        out_verts = self.out_verts[vertex_idx] 

        # Compare outgoing edge vectors wrt changepoint
        # against outgoing edge vectors wrt sampled point,
        # and make sure their angles don't deviate too much
        changepoint_edges = out_verts - orig_vertex
        changepoint_edges_norm = changepoint_edges / np.linalg.norm(changepoint_edges, axis=1)[:, None]
        sample_edges = out_verts - sampled_vertex
        sample_edges_norm = sample_edges / np.linalg.norm(sample_edges, axis=1)[:, None]
        pairwise_dotprod = np.abs(np.sum(changepoint_edges_norm * sample_edges_norm, axis=1))

        dist = np.linalg.norm(sampled_vertex - orig_vertex)
        return (
            dist < self.vertex_radius and
            np.all(pairwise_dotprod > self.max_edge_angular_change_rad)
        )

    def getNeighbours(self, vertex_idx):
        sampled_neighbours = np.zeros((self.n_max_neighbours, 3))
        vert_types = np.zeros(self.n_max_neighbours, dtype=np.int16)
        behaviour_types = -np.ones(self.n_max_neighbours, dtype=np.int16)
        dists = self.pairwise_dists[vertex_idx]

        neighbours = [(idx, dist) for idx, dist in enumerate(dists) if idx != vertex_idx]
        if len(neighbours) > self.n_max_neighbours:
            neighbours = sorted(neighbours, key=itemgetter(1))[:self.n_max_neighbours]
        neighbours = [idx for idx, _ in neighbours]
        count = len(neighbours)

        for i, nidx in enumerate(neighbours):
            sampled_neighbour = None
            while sampled_neighbour is None:
                sampled_neighbour = self.samplePoint(nidx)
            sampled_neighbours[i] = sampled_neighbour
            vert_types[i] = int(self.vertex_types[nidx])
            
        for edge_dst, edge_int, edge_dir in self.out_edges[vertex_idx]:
            if edge_dst in neighbours:
                idx = neighbours.index(edge_dst)
                behaviour_types[idx] = edge_dir * self.n_behaviours + int(edge_int)

        return sampled_neighbours, count, vert_types, behaviour_types


    def generateData(self, num_samples_per_vertex=1):
        sampled_centres = []
        sampled_neighbours = []
        sampled_neighbour_types = []
        sampled_neighbour_counts = []
        sampled_neighbour_behaviours = []
        sampled_map_idxs = []
        sampled_vertex_idxs = []

        for map_idx, env_dir in enumerate(self.env_dirs):
            print(">>> Sampling map: ", env_dir)
            self.setupMap(env_dir)
            
            for vertex_idx in range(len(self.vertex_coords_2d)):
                print("Sampling from vertex: ", vertex_idx)
                count = 0
                while count < num_samples_per_vertex:
                    # Sample a point in the vicinity of the vertex
                    sample = None
                    while sample is None:
                        sample = self.samplePoint(vertex_idx)

                    neighbours, neighbour_count, vert_types, behaviour_types = \
                        self.getNeighbours(vertex_idx)

                    sampled_centres.append(sample)
                    sampled_neighbours.append(neighbours)
                    sampled_neighbour_types.append(vert_types)
                    sampled_neighbour_counts.append(neighbour_count)
                    sampled_neighbour_behaviours.append(behaviour_types)
                    sampled_map_idxs.append(map_idx)
                    sampled_vertex_idxs.append(vertex_idx)
                    count += 1

        perms_dim = np.array([self.n_max_neighbours, self.n_behaviours * self.n_directions])

        np.savez(
            os.path.join(os.getcwd(), self.data_dir, 'preaug_edge_matching_samples.npz'),
            env_dirs=self.env_dirs,
            centres=sampled_centres,
            neighbours=sampled_neighbours,
            neighbour_types=sampled_neighbour_types,
            neighbour_behaviours=sampled_neighbour_behaviours,
            counts=sampled_neighbour_counts,
            perms=np.array([]),
            perms_dim=perms_dim,
            map_idxs=sampled_map_idxs,
            vertex_idxs=sampled_vertex_idxs,
            halfh_halfw=np.array([]),
        )


class EdgeMatchingSampler:
    def __init__(
        self,
        data_dir,
        env_dirs,
        rng,
        im_half_length=100,
        im_half_width=100,
        vertex_radius=0.5,
        K=10,
        n_directions=2,
        n_behaviours=None
    ):
        self.data_dir = data_dir
        self.env_dirs = env_dirs
        self.rng = rng
        self.im_half_length = im_half_length
        self.im_half_width = im_half_width

        self.vertex_radius = vertex_radius
        self.K = K
        self.n_directions = n_directions
        self.n_behaviours = len(Intention) if n_behaviours is None else n_behaviours


    def setupMap(self, env_dir):
        path = os.path.join(os.getcwd(), self.data_dir)

        print("Loading map")
        tmp = np.load(os.path.join(path, env_dir + "_map.npz"))
        self.map = tmp['map']
        self.map_dims = self.map.shape
        self.bounds = tmp['bounds']
        self.res = tmp['res']

        print("Loading graph")
        graph = BehaviourGraph()
        graph.loadGraph(os.path.join(path, env_dir + "_graph.json"))

        # Manually get out edges because the behaviour graph
        # structure does not specify edge_dir in its vertex out_edges
        self.out_edges = [[] for i in range(len(graph.graph_nodes))]
        for edge_src, edge_dst, edge_int, edge_dir in graph.graph_edges:
            self.out_edges[edge_src].append((edge_dst, edge_int, edge_dir))

        self.vertex_types = [label for _, _, label in graph.graph_nodes]
        self.vertex_coords_3d = np.array([pos for _, pos, _ in graph.graph_nodes])
        self.vertex_coords_2d = self.vertex_coords_3d[:, [0, 2]]

        out_vert_coords = [[graph.graph_nodes[dst][1] for dst, _, _ in vert_edges] 
            for vert_edges in self.out_edges]
        self.out_verts_2d = [np.array([[px, py] for px, _, py in points])
            for points in out_vert_coords]


    def convertMap2Pixel(self, point):
        return mapPoint2Pixel(point, self.bounds, self.res)


    def convertPixel2Map(self, px, py):
        return np.array([
            px * self.res + self.bounds[0][0],
            py * self.res - self.bounds[1][2]
        ])


    def insideCrop(self, src_pos, dst_pos):
        src_pos = np.array([src_pos[0], 0, src_pos[1]])
        dst_pos = np.array([dst_pos[0], 0, dst_pos[1]])

        src_px, src_py = self.convertMap2Pixel(src_pos)
        dst_px, dst_py = self.convertMap2Pixel(dst_pos)

        if (
            abs(src_px - dst_px) < self.im_half_width 
            and abs(src_py - dst_py) < self.im_half_length
        ):
            return True
        else:
            return False


    def checkValidSample(self, sample, vertex_idx):
        changepoint = self.vertex_coords_2d[vertex_idx]
        out_verts = self.out_verts_2d[vertex_idx] 

        # Compare outgoing edge vectors wrt changepoint
        # against outgoing edge vectors wrt sampled point,
        # and make sure their angles don't deviate too much
        changepoint_edges = out_verts - changepoint
        changepoint_edges_norm = changepoint_edges / np.linalg.norm(changepoint_edges, axis=1)
        sample_edges = out_verts - sample
        sample_edges_norm = sample_edges / np.linalg.norm(sample_edges, axis=1)
        pairwise_dotprod = np.abs(np.sum(changepoint_edges_norm * sample_edges_norm, axis=1))

        dist = np.linalg.norm(sample - changepoint)
        return (
            dist < self.changepoint_radius and
            np.all(pairwise_dotprod > self.max_edge_angular_change_rad)
        )


    def samplePoint(self, vertex_idx):
        vertex_x, vertex_y = self.vertex_coords_2d[vertex_idx]
        r = np.sqrt(self.rng.random()) * self.vertex_radius
        theta = self.rng.random() * 2 * np.pi
        x = vertex_x + r * np.cos(theta)
        y = vertex_y + r * np.sin(theta)

        return np.array([x, 0, y])

        # sample = np.array([x, y])
        # if self.checkValidSample(sample, vertex_idx):
        #     return np.array([x, 0, y])
        # return None


    def samplePointInsideCrop(self, centre_pos, vertex_pos):
        r = np.sqrt(self.rng.random()) * self.vertex_radius
        theta = self.rng.random() * 2 * np.pi
        x = vertex_pos[0] + r * np.cos(theta)
        y = vertex_pos[1] + r * np.sin(theta)

        if not self.insideCrop(centre_pos, np.array([x, y])):
            return None
        
        return np.array([x, 0, y])


    def truncateRay(self, centre_pos, vertex_pos):
        centre_pos = np.array([centre_pos[0], 0, centre_pos[1]])
        vertex_pos = np.array([vertex_pos[0], 0, vertex_pos[1]])

        cpx, cpy = self.convertMap2Pixel(centre_pos)
        vpx, vpy = self.convertMap2Pixel(vertex_pos)

        tpx = min(cpx + self.im_half_width, max(cpx - self.im_half_width, vpx))
        tpy = min(cpy + self.im_half_length, max(cpy - self.im_half_length, vpy))

        return self.convertPixel2Map(tpx, tpy)


    def getNeighbours(self, sampled_pos, vertex_idx, add_all_edges=False):
        if len(sampled_pos) == 3:
            sampled_pos = np.array([sampled_pos[0], sampled_pos[2]])

        sampled_neighbours = np.zeros((self.K, 3))
        vert_types = np.zeros(self.K, dtype=np.int16)

        if add_all_edges:
            out_verts = [dst for dst, _, _ in self.out_edges[vertex_idx]]
            neighbours_in_crop = [
                (i, np.linalg.norm(pos - sampled_pos)) for i, pos in enumerate(self.vertex_coords_2d)
                if (i != vertex_idx and i not in out_verts and self.insideCrop(sampled_pos, pos))
            ]

            remaining_size = self.K - len(out_verts)
            neighbours_in_crop = (
                sorted(neighbours_in_crop, key=lambda n: n[1])[:remaining_size]
                if len(neighbours_in_crop) > remaining_size else neighbours_in_crop
            )

            i = 0
            for out_vert in out_verts:
                truncated_vert = self.truncateRay(sampled_pos, self.vertex_coords_2d[out_vert])
                sample_neighbour = None
                while sample_neighbour is None:
                    sample_neighbour = self.samplePointInsideCrop(sampled_pos, truncated_vert)
                sampled_neighbours[i] = sample_neighbour
                vert_types[i] = int(self.vertex_types[out_vert])
                i += 1

            for nidx, _ in neighbours_in_crop:
                npoint = self.vertex_coords_2d[nidx]
                sample_neighbour = None
                while sample_neighbour is None:
                    sample_neighbour = self.samplePointInsideCrop(sampled_pos, npoint)
                sampled_neighbours[i] = sample_neighbour
                vert_types[i] = int(self.vertex_types[nidx])
                i += 1

            neighbours = out_verts + [i for i, _ in neighbours_in_crop]

        else:
            neighbours = [
                (i, np.linalg.norm(pos - sampled_pos)) for i, pos in enumerate(self.vertex_coords_2d)
                if i != vertex_idx and self.insideCrop(sampled_pos, pos)
            ]
            neighbours = (
                sorted(neighbours, key=lambda n: n[1])[:self.K]
                if len(neighbours) > self.K else neighbours
            )
            neighbours = [idx for idx, _ in neighbours]

            for i, nidx in enumerate(neighbours):
                npoint = self.vertex_coords_2d[nidx]
                sample_neighbour = None
                while sample_neighbour is None:
                    sample_neighbour = self.samplePointInsideCrop(sampled_pos, npoint)
                sampled_neighbours[i] = sample_neighbour
                vert_types[i] = int(self.vertex_types[nidx])

        count = len(neighbours)
        perms = np.zeros((self.K, self.n_behaviours * self.n_directions))

        for edge_dst, edge_int, edge_dir in self.out_edges[vertex_idx]:
            if edge_dst in neighbours:
                idx = neighbours.index(edge_dst)
                perms[idx, edge_dir * self.n_behaviours + int(edge_int)] = 1.
            elif not add_all_edges:
                print("FOV too small, edge not in crop.")

        return sampled_neighbours, count, perms, vert_types


    def generateData(self, num_samples_per_vertex=30):
        sampled_centres = []
        sampled_neighbours = []
        sampled_neighbour_types = []
        sampled_neighbour_counts = []
        sampled_perms = []
        sampled_map_idxs = []

        for map_idx, env_dir in enumerate(self.env_dirs):
            self.setupMap(env_dir)
            
            for vertex_idx in range(len(self.vertex_coords_2d)):
                print("Sampling from vertex: ", vertex_idx)
                count = 0
                while count < num_samples_per_vertex:
                    # Sample a point in the vicinity of the vertex
                    sample = self.samplePoint(vertex_idx)

                    # If sample is valid, get its nearest neighbours
                    # and outgoing edges and write that as a datum
                    if sample is not None:
                        neighbours, neighbour_count, perms, vert_types = self.getNeighbours(sample, vertex_idx)

                        sampled_centres.append(sample)
                        sampled_neighbours.append(neighbours)
                        sampled_neighbour_types.append(vert_types)
                        sampled_neighbour_counts.append(neighbour_count)
                        sampled_perms.append(perms)
                        sampled_map_idxs.append(map_idx)

                        count += 1
                    print("next: ", count)

        np.savez(
            os.path.join(os.getcwd(), self.data_dir, 'edge_matching_samples.npz'),
            env_dirs=self.env_dirs,
            centres=sampled_centres,
            neighbours=sampled_neighbours,
            neighbour_types=sampled_neighbour_types,
            counts=sampled_neighbour_counts,
            perms=sampled_perms,
            perms_dim=np.array([]),
            behaviour_types=np.array([]),
            map_idxs=sampled_map_idxs,
            halfh_halfw=np.array([self.im_half_length, self.im_half_width])
        )