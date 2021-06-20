import math
import numpy as np
import time
import redis


def zscore(offers_dict, flipped=False):
    """This function implements the computation of the z-score weights for a determinant factor across all offers.
    Inputs:
    - offers_dict :dictionary containing values of a determinant factor. Values are identified by offer identifiers
    as keys
    - flipped: binary value indicating whether resulting weights need to be flipped (i.e. subtracted from 1)
    Outputs:
    - z_score: dictionary containing z-score values for a determinant factor. Values are identified by offer
    identifiers as keys"""

    n = 0
    summation = 0.0
    sum_square = 0.0

    for value in offers_dict.values():
        if value is not None:
            n += 1
            summation += value
            sum_square += value * value

    z_scores = dict()
    if n > 0:
        average = summation / n
        variance = sum_square / n - average * average
        if variance < 1e-7:
            std = 0.0
        else:
            std = math.sqrt(variance)
        for key, value in offers_dict.items():
            if value is not None:
                if std == 0.0:
                    z_scores.setdefault(key, 0.0)
                else:
                    if not flipped:
                        z_scores.setdefault(key, (value - average) / std)
                    else:
                        z_scores.setdefault(key, 1 - (value - average) / std)
    return z_scores


def check_if_equal(list_1, list_2):
    """ Check if two lists are equal"""
    if list_1 == list_2:
        return 1
    else:
        return 0


# ROD weights (up to 10 features per category)
feature_2 = [0.6932, 0.3068]
feature_3 = [0.5232, 0.3240, 0.1528]
feature_4 = [0.4180, 0.2986, 0.1912, 0.0922]
feature_5 = [0.3471, 0.2686, 0.1955, 0.1269, 0.0619]
feature_6 = [0.2966, 0.2410, 0.1884, 0.1387, 0.0908, 0.0445]
feature_7 = [0.2590, 0.2174, 0.1781, 0.1406, 0.1038, 0.0679, 0.0334]
feature_8 = [0.2292, 0.1977, 0.1672, 0.1375, 0.1084, 0.0805, 0.0531, 0.0263]
feature_9 = [0.2058, 0.1808, 0.1565, 0.1332, 0.1095, 0.0867, 0.0644, 0.0425, 0.0211]
feature_10 = [0.1867, 0.1667, 0.1466, 0.1271, 0.1081, 0.0893, 0.0709, 0.0527, 0.0349, 0.0173]
ROD = {2: feature_2, 3: feature_3, 4: feature_4, 5: feature_5, 6: feature_6, 7: feature_7, 8: feature_8, 9: feature_9,
       10: feature_10}


def rod_aggregation(features_dict):
    """This function aggregate the values of the different features within a category to compute the final
    category score. The ROD weights are used"""
    scores_dict = dict()
    for key, list_values in features_dict.items():
        rod_weights = np.array(ROD[len(list_values)])
        category_score = np.dot(np.array(list_values), rod_weights)
        scores_dict.setdefault(key, category_score)
    return scores_dict


def extract_data_from_cache(pa_cache, pa_request_id, pa_request_level_items,
                            pa_offer_level_items, pa_tripleg_level_items):
    """This is a function to read specific data from the cache.
    Inputs:
    - pa_cache: cache identifier
    - pa_request_id: request id for which the data should be extracted from cache
    - pa_request_level_items: list of attributes at the request level that should be extracted from the cache
    - pa_offer_level_items: list of attributes at the offer level that should be extracted from the cache
    - pa_tripleg_level_items: list of attributes at the trip leg level that should be extracted from the cache
    Outputs:
    - output_request_level_items: dictionary containing values of requested attributes at the request level
    - output_offer_level_items - dictionary containing values of requested attributed at the offer level
    - output_tripleg_level_items - dictionary containing values of requested attributed at the trip leg level"""
    output_request_level_items = dict()
    output_offer_level_items = dict()
    output_tripleg_level_items = dict()

    # request level information
    if len(pa_request_level_items) > 0:
        for request_level_item in pa_request_level_items:
            item = pa_cache.get('{}:{}'.format(pa_request_id, request_level_item))
            output_request_level_items['{}'.format(request_level_item)] = item

    offer_ids = pa_cache.lrange('{}:offers'.format(pa_request_id), 0, -1)
    output_offer_level_items["offer_ids"] = offer_ids
    pipe = pa_cache.pipeline()
    if offer_ids is not None:
        for offer in offer_ids:
            output_offer_level_items[offer] = {}
            for offer_level_item in pa_offer_level_items:
                # assembly key for offer level
                temp_key = "{}:{}:{}".format(pa_request_id, offer, offer_level_item)
                # extract offer level data from cache
                if (offer_level_item == "bookable_total") or (offer_level_item == "complete_total"):
                    pipe.hgetall(temp_key)
                else:
                    pipe.get(temp_key)
            # extract information at the trip leg level
            output_tripleg_level_items[offer] = {}
            if len(pa_tripleg_level_items) > 0:
                temp_key = "{}:{}:legs".format(pa_request_id, offer)
                tripleg_ids = pa_cache.lrange(temp_key, 0, -1)
                output_tripleg_level_items[offer]["triplegs"] = tripleg_ids
                for tripleg_id in tripleg_ids:
                    output_tripleg_level_items[offer][tripleg_id] = {}
                    for tripleg_level_item in pa_tripleg_level_items:
                        temp_key = "{}:{}:{}:{}".format(pa_request_id, offer, tripleg_id, tripleg_level_item)
                        pipe.get(temp_key)
        temp_data = pipe.execute()
        index = 0
        for offer in offer_ids:
            for offer_level_item in pa_offer_level_items:
                output_offer_level_items[offer][offer_level_item] = temp_data[index]
                index += 1
            if len(pa_tripleg_level_items) > 0:
                tripleg_ids = output_tripleg_level_items[offer]["triplegs"]
                for tripleg_id in tripleg_ids:
                    for tripleg_level_item in pa_tripleg_level_items:
                        output_tripleg_level_items[offer][tripleg_id][tripleg_level_item] = temp_data[index]
                        index += 1

    return output_request_level_items, output_offer_level_items, output_tripleg_level_items


def read_data_from_cache_wrapper(pa_cache, pa_request_id, pa_request_level_items,
                                 pa_offer_level_items, pa_tripleg_level_items):
    """Wrapper procedure for reading operation from cache used by feature collectors. Wrapper ensures repeated reading
    attempts when reading from cache is failing. After a certain number of unsuccessful attempts an error is raised."""
    retries = 5

    while True:
        try:
            return extract_data_from_cache(pa_cache, pa_request_id, pa_request_level_items,
                                           pa_offer_level_items, pa_tripleg_level_items)
        except redis.exceptions.ConnectionError as exc:
            print("Reading from cache by a feature collector failed. Retries remaining: {}".format(retries))
            if retries == 0:
                raise exc
            retries -= 1
            time.sleep(0.1)
