import numpy as np
from categories.utils.utils import zscore


def compute_normalized_co2_km(co2_dict, dist_dict):
    """
    This function returns the normalized co2/km of each alternative
    The inputs are dictionaries:
     - co2_dict: contains the co2 emissions of each alternative (in kg)
     - dist_dict: contains the distance covered by each offer (in km)
     """
    co2_per_km = dict()
    for co2, distance in zip(co2_dict.items(), dist_dict.values()):
        key = co2[0]
        if distance != 0:
            if co2[1] is not None:
                co2_value = float(co2[1])
            else:
                co2_value = 0.0
            co2_per_km.setdefault(key, co2_value/distance)
        else:
            co2_per_km.setdefault(key, 0.0)
    normalized_co2_km = zscore(co2_per_km, flipped=True)
    return normalized_co2_km
