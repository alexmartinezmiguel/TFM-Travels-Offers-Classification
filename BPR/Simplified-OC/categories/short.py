import numpy as np
from categories.utils.utils import zscore


def harvesine_distance(coord_ini, coord_end):
    """
    This function computes the Harvesine distance between two given coordinates
    """
    lat_ini = coord_ini[0] * np.pi / 180  # rad
    long_ini = coord_ini[1] * np.pi / 180  # rad
    lat_end = coord_end[0] * np.pi / 180  # rad
    long_end = coord_end[1] * np.pi / 180  # rad
    R = 6371000  # m
    a = (np.sin((lat_end - lat_ini) / 2)) ** 2 + np.cos(lat_ini) * np.cos(lat_end) * (
        np.sin((long_end - long_ini) / 2)) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def compute_normalized_distance(distances_dict):
    """
    This function returns the normalized distance of each alternative
    - Input: dictionary containing distance covered by each one of the alternatives in km
    """
    normalized_values = zscore(distances_dict, flipped=True)
    return normalized_values


def compute_normalized_num_stops(num_stops_dict):
    """
    This function returns the normalized number of stops of each alternative
    - Input: dictionary containing the number of stops of each one of the alternatives
    """
    normalized_values = zscore(num_stops_dict, flipped=True)
    return normalized_values
