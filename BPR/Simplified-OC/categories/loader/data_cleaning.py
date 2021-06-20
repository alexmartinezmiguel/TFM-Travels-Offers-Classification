import os
import numpy as np


def filter_trips(routerank_dataset):
    """-Input: Original routerank dataset
    This function filters the trips with no alternatives and the trips not present in the MoTiv dataset"""
    # first, filter the trips with no alternatives
    print('Filtering trips with no alternatives...')
    tripid_no_alternatives = []
    for trip in routerank_dataset:
        if len(trip['alternatives']) == 0:
            tripid_no_alternatives.append(trip['tripId'])

    routerank_alternatives = []
    for trip in routerank_dataset:
        if trip['tripId'] not in tripid_no_alternatives:
            routerank_alternatives.append(trip)

    # secondly, filter the trips which are not present in the MoTiv dataset (original trips)
    print('Filtering trips not present in the MoTiv dataset...')
    base_dir = 'categories/loader'
    filename_motiv_request_id = 'unique_request_id.txt'
    abs_file_requests_id = os.path.join(base_dir, filename_motiv_request_id)
    unique_requests_id = open(abs_file_requests_id, 'r').read().split('\n')

    final_alternatives = []
    for trip in routerank_alternatives:
        if trip['tripId'] in unique_requests_id:
            final_alternatives.append(trip)

    return final_alternatives


def combine_alternatives(legs):
    """This function returns all possible combinations of leg alternatives to form
        trip alternatives. The input is a list containing the leg alternatives from
        the routeRank dataset"""

    trip_dict = dict()
    trip_dict.setdefault('from', legs[0]['from'])
    trip_dict.setdefault('to', legs[-1]['to'])
    trip_dict.setdefault('date', legs[0]['date'])
    trip_dict.setdefault('tripId', legs[0]['tripId'])
    trip_dict.setdefault('places', {})
    number_alternatives_trip = []
    for trip in legs:
        trip_dict['places'].update(trip['places'])
        number_alternatives_trip.append(len(trip['alternatives']))
    number_alternatives_trip.append(1)
    number_alternatives = np.prod(number_alternatives_trip)
    l = [[] for i in range(number_alternatives)]
    trip_dict.setdefault('alternatives', l)
    for j in range(len(legs)):
        k = np.prod(number_alternatives_trip[j + 1:])
        v = 0
        for alternative in legs[j]['alternatives']:
            for i in range(number_alternatives):
                if i % np.prod(number_alternatives_trip[j:]) == 0:
                    for m in range(k):
                        trip_dict['alternatives'][i + m + v].append(alternative)
            v += k
    return trip_dict
