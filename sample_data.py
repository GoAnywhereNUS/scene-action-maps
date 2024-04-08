import os
import numpy as np
import argparse

from data_samplers import ChangepointSampler, EdgeMatchingSampler, ReducedEdgeMatchingSampler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cp_num_changepoint_samples", type=int, help="Number of changepoint samples",
        default=60)
    parser.add_argument("--cp_neg_to_pos_sample_ratio", type=float, help="Ratio of negative to positive samples",
        default=1.2)
    parser.add_argument("--cp_changepoint_radius", type=int, help="Changepoint sampling radius",
        default=0.25) # For satellite maps
    parser.add_argument("--em_im_half_length", type=int, help="Half length of image crop",
        default=100)
    parser.add_argument("--em_im_half_width", type=int, help="Half width of image crop",
        default=100)
    parser.add_argument("--em_vertex_radius", type=float, help="Vertex sampling radius",
        default=0.25) # For satellite maps
    parser.add_argument("--em_num_vertex_samples", type=int, help="Number of vertex samples",
        default=50)
    parser.add_argument("--map_type", type=str, help="Map type", default="satmap")
    parser.add_argument("--env_dirs", type=str, nargs='+', help='List of maps to process')
    args = parser.parse_args()

    env_dirs = args.env_dirs if args.env_dirs else ["TuasSouth1", "TuasSouth3", "TuasSouth4"] # For satellite maps

    cp_sampler = ChangepointSampler(
        args.map_type,
        env_dirs=env_dirs,
        rng=np.random.default_rng(0),
        changepoint_radius=args.cp_changepoint_radius
    )

    print("Generate changepoint data")
    cp_sampler.generateData(
        num_pos_samples_per_changepoint=args.cp_num_changepoint_samples,
        neg_to_pos_sample_ratio=args.cp_neg_to_pos_sample_ratio
    )

    print("Generate edge matching data")

    em_sampler = ReducedEdgeMatchingSampler(
        args.map_type,
        env_dirs=env_dirs,
        rng=np.random.default_rng(0),
        vertex_radius=args.em_vertex_radius
    )

    print("Generate edge matching data")
    em_sampler.generateData(args.em_num_vertex_samples)
