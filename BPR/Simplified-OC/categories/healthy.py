import numpy as np
from categories.utils.utils import zscore


def compute_normalized_bike_walk_dist(bike_walk_dict, dist_dict, flipped=False):
    """
    This function returns the normalized walking/bike distance of each alternative
    The inputs are dictionaries:
     - bike_walk_dict: contains the distance covered by walk/bike (in km)
     - dist_dict: contains the distance covered by each offer (in km)"""
    fraction_bike_walk = dict()
    for bw_dist, distance in zip(bike_walk_dict.items(), dist_dict.values()):
        key = bw_dist[0]
        if distance != 0:
            fraction_bike_walk.setdefault(key, bw_dist[1] / distance)
        else:
            fraction_bike_walk.setdefault(key, 0.0)
    normalized_fraction_bike_walk = zscore(fraction_bike_walk, flipped)
    return normalized_fraction_bike_walk
