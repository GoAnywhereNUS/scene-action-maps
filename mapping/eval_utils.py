import numpy as np

from structures import BehaviourGraph
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist

class GraphCompare:
    def __init__(self, max_dist_threshold):
        self.max_dist_threshold = max_dist_threshold
        self.gt_graph = None
        self.test_graph = None
        self.results = {}

    def _precision(self, tp_count, fp_count):
        return float(tp_count) / float(tp_count + fp_count)

    def _recall(self, tp_count, fn_count):
        return float(tp_count) / float(tp_count + fn_count)

    def _f1score(self, tp_count, fp_count, fn_count):
        precision = self._precision(tp_count, fp_count)
        recall = self._recall(tp_count, fn_count)
        assert (precision + recall) != 0.0
        return 2 * (precision * recall) / (precision + recall)

    def setGroundTruth(self, graph):
        assert isinstance(graph, BehaviourGraph)
        self.gt_graph = graph

    def setTestGraph(self, graph):
        assert isinstance(graph, BehaviourGraph)
        self.test_graph = graph

    def getChangepointStats(self):
        # Solve linear assignment problem to find closest 1-1 matches
        # between the changepoints of the ground truth and test graphs
        gt_cp_locations = np.array([
            pos for _, pos, label in self.gt_graph.graph_nodes if not label
        ])[:, [0, 2]]
        test_cp_locations = np.array([
            pos for _, pos, label in self.test_graph.graph_nodes if not label
        ])[:, [0, 2]]
        costs = cdist(gt_cp_locations, test_cp_locations)
        gt_idxs, test_idxs = linear_sum_assignment(costs)

        # Filter out invalid matches (too far away)
        print(len(gt_idxs), len(test_idxs))
        paired_idxs = [
            (gt_idx, test_idx) for gt_idx, test_idx in zip(gt_idxs, test_idxs)
            if costs[gt_idx, test_idx] < self.max_dist_threshold
        ]

        # Count TP/FP/FN
        tp_pairs = paired_idxs
        fp_idxs = set(range(len(test_cp_locations))).difference({test_idx for _, test_idx in paired_idxs})
        fn_idxs = set(range(len(gt_cp_locations))).difference({gt_idx for gt_idx, test_idx in paired_idxs})
        
        precision = self._precision(len(tp_pairs), len(fp_idxs))
        recall = self._recall(len(tp_pairs), len(fn_idxs))
        f1_score = self._f1score(len(tp_pairs), len(fp_idxs), len(fn_idxs))

        print("=== Changepoint stats ===")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)

        print("\n>>> Counts")
        print("Num GT nodes: ", len(gt_idxs), ", Num test nodes: ", len(test_idxs))
        print("TP: ", len(tp_pairs))
        print("FP: ", len(fp_idxs))
        print("FN: ", len(fn_idxs))

        self.results['cp_precision'] = precision
        self.results['cp_recall'] = recall
        self.results['cp_f1_score'] = f1_score
        self.results['cp_tp_pairs'] = tp_pairs
        self.results['cp_fp_idxs'] = fp_idxs
        self.results['cp_fn_idxs'] = fn_idxs
        

    def getBehaviourEdgeStats(self):
        # Compute absolute TP/FP/FN for all edges directly
        gt_edges = {
            (src, dst, i) for src, dst, i, _ in self.gt_graph.graph_edges
        }
        test_edges = {
            (src, dst, i) for src, dst, i, _ in self.test_graph.graph_edges
        }

        tp_edges = gt_edges.intersection(test_edges)
        fp_edges = test_edges.difference(gt_edges)
        fn_edges = gt_edges.difference(test_edges)

        precision = self._precision(len(tp_edges), len(fp_edges))
        recall = self._recall(len(tp_edges), len(fn_edges))
        f1_score = self._f1score(len(tp_edges), len(fp_edges), len(fn_edges))

        # Compute a relaxed TP/FP/FN using edges without accounting for intention labels
        gt_edges_noint = {(src, dst) for src, dst, _, _ in self.gt_graph.graph_edges}
        test_edges_noint = {(src, dst) for src, dst, _, _ in self.test_graph.graph_edges}

        tp_edges_noint = gt_edges_noint.intersection(test_edges_noint)
        fp_edges_noint = test_edges_noint.difference(gt_edges_noint)
        fn_edges_noint = gt_edges_noint.difference(test_edges_noint)

        precision_noint = self._precision(len(tp_edges_noint), len(fp_edges_noint))
        recall_noint = self._recall(len(tp_edges_noint), len(fn_edges_noint))
        f1_score_noint = self._f1score(len(tp_edges_noint), len(fp_edges_noint), len(fn_edges_noint))

        print("=== Behaviour edge stats ===")
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1 score: ", f1_score)

        print("\n>>> Intentions not included")
        print("Precision: ", precision_noint)
        print("Recall: ", recall_noint)
        print("F1 score: ", f1_score_noint)

        print("\n>>> Counts")
        print("No. of TP edges (absolute): ", len(tp_edges))
        print("No. of FP edges (absolute): ", len(fp_edges))
        print("No. of FN edges (absolute): ", len(fn_edges))
        print("No. of TP edges (no intention): ", len(tp_edges_noint))
        print("No. of FP edges (no intention): ", len(fp_edges_noint))
        print("No. of FN edges (no intention): ", len(fn_edges_noint))
        print("No. of correctly matched edges with wrong intention: ", len(tp_edges_noint) - len(tp_edges))

        self.results['em_tp_edges'] = tp_edges
        self.results['em_fp_edges'] = fp_edges
        self.results['em_fn_edges'] = fn_edges
        self.results['em_tp_edges_noint'] = tp_edges_noint
        self.results['em_fp_edges_noint'] = fp_edges_noint
        self.results['em_fn_edges_noint'] = fn_edges_noint
        self.results['em_precision'] = precision
        self.results['em_recall'] = recall
        self.results['em_f1_score'] = f1_score
        self.results['em_precision_noint'] = precision_noint
        self.results['em_recall_noint'] = recall_noint
        self.results['em_f1_score'] = f1_score_noint
        
    def getResults(self):
        return self.results

    def getStats(self, results):
        cp_tp = sum([len(res['cp_tp_pairs']) for res in results])
        cp_fp = sum([len(res['cp_fp_idxs']) for res in results])
        cp_fn = sum([len(res['cp_fn_idxs']) for res in results])
        em_tp = sum([len(res['em_tp_edges']) for res in results])
        em_fp = sum([len(res['em_fp_edges']) for res in results])
        em_fn = sum([len(res['em_fn_edges']) for res in results])
        em_tp_noint = sum([len(res['em_tp_edges_noint']) for res in results])
        em_fp_noint = sum([len(res['em_fp_edges_noint']) for res in results])
        em_fn_noint = sum([len(res['em_fn_edges_noint']) for res in results])

        print("=== Changepoint stats ===")
        print("Precision: ", self._precision(cp_tp, cp_fp))
        print("Recall: ", self._recall(cp_tp, cp_fn))
        print("F1 score: ", self._f1score(cp_tp, cp_fp, cp_fn))
        print("\n")

        print("=== Behaviour edge stats ===")
        print("Precision: ", self._precision(em_tp, em_fp))
        print("Recall: ", self._recall(em_tp, em_fn))
        print("F1 score: ", self._f1score(em_tp, em_fp, em_fn))

        print("\n>>> Intentions not included")
        print("Precision: ", self._precision(em_tp_noint, em_fp_noint))
        print("Recall: ", self._recall(em_tp_noint, em_fn_noint))
        print("F1 score: ", self._f1score(em_tp_noint, em_fp_noint, em_fn_noint))

    def saveStats(self):
        np.savez(
            "eval_stats.npz",
            cp_tp_pairs=self.results['cp_tp_pairs'],
            cp_fp_idxs=self.results['cp_fp_idxs'],
            cp_fn_idxs=self.results['cp_fn_idxs'],
            cp_precision=self.results['cp_precision'],
            cp_recall=self.results['cp_recall'],
            cp_f1_score=self.results['cp_f1_score'],
            em_tp_edges=self.results['em_tp_edges'],
            em_fp_edges=self.results['em_fp_edges'],
            em_fn_edges=self.results['em_fn_edges'],
            em_precision=self.results['em_precision'],
            em_recall=self.results['em_recall'],
            em_f1_score=self.results['em_f1_score']
        )