import os
import json
import heapq
import numpy as np
import bisect
from enum import IntEnum
from copy import deepcopy

class Node(IntEnum):
    CHANGEPOINT = 0
    PLACE = 1

class Intention(IntEnum):
    LEFT = 0
    FORWARD = 1
    RIGHT = 2

def flipIntention(intention):
    if intention == Intention.LEFT:
        return Intention.RIGHT
    elif intention == Intention.FORWARD:
        return Intention.FORWARD
    elif intention == Intention.RIGHT:
        return Intention.LEFT
    else:
        raise NotImplementedError

class BehaviourGraph:
    def __init__(self, scene_dir=""):
        self.scene_dir = scene_dir
        self.graph_nodes = []
        self.graph_edges = []

        self.vertices = []
        self.graph_edges_map = {}
        self.cumulative_edge_dists = []

        self.rng = None

    # I/O
    def writeGraph(self, outfile=None):
        basename = os.path.basename(self.scene_dir)
        dirname = os.path.dirname(self.scene_dir)
        scenename = basename.split(".")[0]
        filename = scenename + "_graph.json"

        if outfile is None:
            outfile = os.path.join(dirname, filename)

        flattened_graph_nodes = [(*pix, *pos) for pix, pos, _ in self.graph_nodes]
        pos_graph_nodes = list(map(lambda tup: tuple(map(float, tup)), flattened_graph_nodes))
        labels_graph_nodes = [place for _, _, place in self.graph_nodes]
        graph_edges = [tuple(map(int, tup)) for tup in self.graph_edges]
        data = {'scene_name': scenename,
                'scene_dir': self.scene_dir,
                'graph_node_pos': pos_graph_nodes,
                'graph_node_labels': labels_graph_nodes,
                'graph_edges': graph_edges }

        print("Writing data to ", outfile)
        with open(outfile, 'w') as f:
            json.dump(data, f)

    def loadGraph(self, graph_dir):
        print("Loading data from ", graph_dir)
        with open(graph_dir, 'r') as f:
            data = json.load(f)

            print("Setting scene as ", data['scene_name'])
            pos = data['graph_node_pos']
            labels = data['graph_node_labels']
            self.scene_dir = data['scene_dir']
            self.graph_nodes = [(np.array([p[0], p[1]]), [p[2], p[3], p[4]], l) for p, l in zip(pos, labels)]
            self.graph_edges = [(e[0], e[1], Intention(e[2]), e[3]) for e in data['graph_edges']]

    def writeRawData2Graph(self, scene_dir, nodes, edges, output_name=None):
        self.scene_dir = scene_dir
        self.graph_nodes = nodes
        self.graph_edges = edges
        if output_name is None:
            output_name = 'predicted_graph.json'
        self.writeGraph(outfile=os.path.join(os.getcwd(), output_name))

    def inTriangle(self, point, v1, v2, v3):
        # Extract the vertices of the triangle
        x1, _, y1 = v1
        x2, _, y2 = v2
        x3, _, y3 = v3

        # Compute the vectors representing the sides of the triangle
        v1 = (x3-x1, y3-y1)
        v2 = (x2-x1, y2-y1)
        v3 = (point[0]-x1, point[2]-y1)

        # Compute the dot products of the vectors
        dot11 = v1[0]*v1[0] + v1[1]*v1[1]
        dot12 = v1[0]*v2[0] + v1[1]*v2[1]
        dot13 = v1[0]*v3[0] + v1[1]*v3[1]
        dot22 = v2[0]*v2[0] + v2[1]*v2[1]
        dot23 = v2[0]*v3[0] + v2[1]*v3[1]

        # Compute the barycentric coordinates of the point with respect to the triangle
        inv_denom = 1.0 / (dot11 * dot22 - dot12 * dot12)
        u = (dot22 * dot13 - dot12 * dot23) * inv_denom
        v = (dot11 * dot23 - dot12 * dot13) * inv_denom

        # Check if the point is inside the triangle
        return (u > 0) and (v > 0) and (u + v < 1)


    def graphify(self, scene_dir, nodes, edges, 
        write=False, merge_dist_threshold=0.5):
        edges_map = {(src, dst):(intent, idx) for src, dst, intent, idx in edges}
        node_out_edges = [[] for _ in nodes]
        for src, dst, i, d in edges:
            node_out_edges[src].append((dst, i, d))

        # Enforce that all edges go in both directions
        unmatched_edges = []
        for src, dst in edges_map.keys():
            if (dst, src) not in edges_map:
                intent, idx = edges_map[(src, dst)]
                unmatched_edges.append((src, dst, Intention(intent), idx))

        def matchIntention(i):
            if i == Intention.FORWARD:
                return Intention.FORWARD
            elif i == Intention.LEFT:
                return Intention.RIGHT
            elif i == Intention.RIGHT:
                return Intention.LEFT
            else:
                raise AssertionError("Unknown intention given! ", str(i))

        def isDirectionConsistent(next_src, next_dst, d, out_edges):
            _, next_src_coords, _ = nodes[next_src]
            _, next_dst_coords, _ = nodes[next_dst]
            next_src_coords, next_dst_coords = np.array(next_src_coords), np.array(next_dst_coords)
            next_dir = next_dst_coords - next_src_coords
            next_dir_norm = next_dir / np.linalg.norm(next_dir)
            same_dirs = [nodes[edst][1] for edst, _, edir in out_edges if edir == d]
            
            if len(same_dirs) == 0:
                # No other edges in same direction, so vacuously true 
                # that direction is consistent
                return True

            same_dir_coords = np.array(same_dirs) - next_src_coords
            same_dir_norm = same_dir_coords / np.expand_dims(np.linalg.norm(same_dir_coords, axis=1), axis=1)
            dotprod = np.dot(same_dir_norm, next_dir_norm)
            return np.all(dotprod > 0)

        def getDirection(next_src, next_dst, i):
            out_edges = node_out_edges[next_src]
            i_out_edges = [edge for edge in out_edges if edge[1] == i]

            if len(i_out_edges) == 2:
                print("Cannot add more edges of type", i, "to", next_src, "! ", i_out_edges)
                return None

            elif len(i_out_edges) < 2:
                used_dirs = [used_dir for  _, _, used_dir in i_out_edges]
                available_dirs = [d for d in range(2) if d not in used_dirs]
                
                for d in available_dirs:
                    if isDirectionConsistent(next_src, next_dst, d, out_edges):
                        return d

                print("Available directions fail consistency check")
                return None

            else:
                raise AssertionError("Too many edges with intention ", i, "!")
            
        matched_edges = []
        for orig_src, orig_dst, i, d in unmatched_edges:
            matched_i = matchIntention(i)
            matched_d = getDirection(orig_dst, orig_src, i)
            if matched_i is not None and matched_d is not None:
                matched_edges.append((orig_dst, orig_src, matched_i, matched_d))

        print("Adding edges: ", matched_edges)
        edges += matched_edges
        for src, dst, i, d in matched_edges:
            edges_map[(src, dst)] = (i, d)
            node_out_edges[src].append((dst, i, d))

        # Remove skips (look two nodes ahead only)
        removed_edges = []
        for src, first_edges in enumerate(node_out_edges):
            first_dsts = [(dst, i) for dst, i, _ in first_edges]
            second_dsts = [(dst2, i1, i2) for dst1, i1, _ in first_edges for dst2, i2, _ in node_out_edges[dst1]]
            for dst2, i1, i2 in second_dsts:
                if (dst2, i2) in first_dsts:
                    # Flag the edge for removal
                    node_out_edges[src] = [(dst, i, d) for dst, i, d in node_out_edges[src] if dst != dst2]
                    node_out_edges[dst2] = [(n, i, d) for n, i, d in node_out_edges[dst2] if n != src]
                    removed_edges.append((src, dst2))
                    removed_edges.append((dst2, src))

                    # Check before removing, because the matched edge may have already been
                    # removed. Some edges may also not be matched due to failing checks.
                    if (src, dst2) in edges_map.keys():
                        edges_map.pop((src, dst2))
                    if (dst2, src) in edges_map:
                        edges_map.pop((dst2, src))
                        
        print("Removing edges: ", removed_edges)
        edges = [(src, dst, i, d) for src, dst, i, d in edges if (src, dst) not in removed_edges]

        # Find triangular cycles
        removed_edges = []
        added_edges = []
        for (src, dst), (i, d) in edges_map.items():
            src_neighbours = {n for n, _, _ in node_out_edges[src]}
            dst_neighbours = {n for n, _, _ in node_out_edges[dst]}
            common_neighbours = src_neighbours.intersection(dst_neighbours)
            _, srcpt, _ = nodes[src]
            _, dstpt, _ = nodes[dst]

            for neighbour in common_neighbours:
                inside_nodes = []
                _, npt, _ = nodes[neighbour]
                for idx, (_, p, _) in enumerate(nodes):
                    if self.inTriangle(p, srcpt, dstpt, npt):
                        inside_nodes.append(idx)    

                for node in inside_nodes:
                    if node not in src_neighbours:                        
                        removed_edges.append((src, dst))
                        removed_edges.append((dst, src))
                        removed_edges.append((src, neighbour))
                        removed_edges.append((neighbour, src))
                        added_edges.append((src, node, Intention.FORWARD, 0))
                        added_edges.append((node, src, Intention.FORWARD, 0))

        print("Removing edges: ", removed_edges)
        added_edges = list(set(added_edges))
        edges = [(src, dst, i, d) for src, dst, i, d in edges if (src, dst) not in removed_edges]

        print("Adding back new edges: ", added_edges)
        edges += added_edges

        if write:
            self.writeRawData2Graph(scene_dir, nodes, edges)
        else:
            self.scene_dir = scene_dir
            self.nodes = nodes
            self.edges = edges


    # Sampling and graph search
    def initialise(self, random_seed=0):
        self.vertices = [{
            'coord': np.array(pos),
            'place_node': place,
            'out_edges': []
            } for _, pos, place in self.graph_nodes]

        edge_dists = [
            np.linalg.norm(self.vertices[dst]['coord'] - self.vertices[src]['coord'])
            for src, dst, _, _ in self.graph_edges
        ]
        self.cumulative_edge_dists = np.cumsum(np.array(edge_dists))

        for (edge_src, edge_dst, edge_int, _), dist in zip(self.graph_edges, edge_dists):
            self.vertices[edge_src]['out_edges'].append(
                (edge_dst, dist, edge_int)
            )

        for edge_idx, (edge_src, edge_dst, edge_int, _) in enumerate(self.graph_edges):
            if (edge_src, edge_dst) in self.graph_edges_map.keys():
                print(edge_src, edge_dst, edge_idx)
                raise Exception("Duplicate edges in graph!")
            self.graph_edges_map[(edge_src, edge_dst)] = (edge_int, edge_idx)

        # Sanity check correctness of graph -- make sure all edges are reversible,
        # i.e. for any edge (src, dst), there also exists the corresponding edge
        # (dst, src)
        for edge_idx, (edge_src, edge_dst, _, _) in enumerate(self.graph_edges):
            if (edge_dst, edge_src) not in self.graph_edges_map.keys():
                print(edge_src, edge_dst, edge_idx)
                raise Exception("Edge not reversible!")

        self.rng = np.random.default_rng(random_seed)


    def dijkstra(self, src, dst):
        dists = [0 if idx == src else np.inf for idx in range(len(self.vertices))]
        prevs = [None for _ in range(len(self.vertices))]
        pq = [(dist, idx) for idx, dist in enumerate(dists)]
        heapq.heapify(pq)

        while len(pq) > 0:
            min_dist, min_idx = pq[0]
            if min_idx == dst:
                pointer = dst
                path = []
                coords = []
                labels = []
                while pointer is not None:
                    path.append(pointer)
                    coords.append(self.vertices[pointer]['coord'])
                    tup = prevs[pointer]
                    if tup is not None:
                        pointer, intention = tup
                        labels.append(intention)
                    else:
                        pointer = None

                path.reverse()
                coords.reverse()
                labels.reverse()

                return path, np.array(coords), labels
                    
            heapq.heappop(pq)

            for nidx, nlen, intention in self.vertices[min_idx]['out_edges']:
                alt = dists[min_idx] + nlen
                if alt < dists[nidx]:
                    dists[nidx] = alt
                    prevs[nidx] = (min_idx, intention)
                    pidx = next(i for i, elem in enumerate(pq) if elem[1] == nidx)
                    pq[pidx] = (alt, nidx)

            heapq.heapify(pq)

        return None

    def sampleEdge(self):
        if len(self.cumulative_edge_dists) < 1:
            return None

        # sampled_dist = np.random.random() * self.cumulative_edge_dists[-1]
        sampled_dist = self.rng.random() * self.cumulative_edge_dists[-1]
        upper_idx = bisect.bisect_right(self.cumulative_edge_dists, sampled_dist)
        sampled_edge_dist_norm = (
            sampled_dist / self.cumulative_edge_dists[upper_idx] if upper_idx == 0 else
            (sampled_dist - self.cumulative_edge_dists[upper_idx - 1]) / (self.cumulative_edge_dists[upper_idx] - self.cumulative_edge_dists[upper_idx - 1])
        )

        edge_src, edge_dst, _, _ = self.graph_edges[upper_idx]
        src_point = self.vertices[edge_src]['coord']
        dst_point = self.vertices[edge_dst]['coord']
        sampled_point = src_point + (dst_point - src_point) * sampled_edge_dist_norm

        return sampled_point, edge_src, edge_dst