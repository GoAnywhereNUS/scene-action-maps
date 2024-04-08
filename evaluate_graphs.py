import os
import numpy as np
from eval_utils import GraphCompare
from structures import BehaviourGraph

if __name__ == "__main__":
    envs = [
        # ('AS6_L2', 'v2'),
        # ('COM2_L4', 'v2'),
        # ('COM1_L1', 'v2'),
        # ('COM1_Basement', 'v1'),
        ('TuasSouth2', 'v1'),
    ]
    max_changepoint_dist_threshold = 2.0 # For hand drawn, floorplans
    max_changepoint_dist_threshold = 5.0 # For satellite map

    map_dir = "maps/satmap"
    results_dir = "results"

    map_type_strs = {'floorplans': 'floor_', 'satmap': 'satmap_', 'hand': 'hand_'}
    map_type_str = ''
    for k, v in map_type_strs.items():
        if k in map_dir:
            map_type_str = v

    stats = []
    for env, ver in envs:
        gt_path = os.path.join(map_dir, env + '_graph.json')
        test_path = os.path.join(results_dir, map_type_str + env.lower() + '_' + ver)

        predicted_cp_path = os.path.join(test_path, "predicted_changepoints.json")
        predicted_edges_path = os.path.join(test_path, "predicted_edges.json")

        comp = GraphCompare(max_dist_threshold=max_changepoint_dist_threshold)

        # Import ground truth
        gt_graph = BehaviourGraph()
        gt_graph.loadGraph(gt_path)
        comp.setGroundTruth(gt_graph)

        # Import and test predicted changepoints
        cp_graph = BehaviourGraph()
        cp_graph.loadGraph(predicted_cp_path)
        comp.setTestGraph(cp_graph)
        comp.getChangepointStats()

        # Import and test predicted edges
        edges_graph = BehaviourGraph()
        edges_graph.loadGraph(predicted_edges_path)
        comp.setTestGraph(edges_graph)
        comp.getBehaviourEdgeStats()

        # Save results
        stats.append(comp.getResults())

    print(">>>>>>>> Overall <<<<<<<<<")
    comp.getStats(stats)
