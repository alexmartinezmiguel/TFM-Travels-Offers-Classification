import configparser as cp
import redis
import geojson
import json
import numpy as np
import os
import pandas as pd

from categories.utils.utils import read_data_from_cache_wrapper
from categories.quick import compute_duration, compute_single_duration
from categories.cheap import compute_normalized_complete_total
from categories.co2 import compute_normalized_co2_km
from categories.healthy import compute_normalized_bike_walk_dist
from categories.utils.utils import rod_aggregation, zscore, check_if_equal

service_name = 'categorizer'

# config
config = cp.ConfigParser()
config.read(f'{service_name}.conf')

cache = redis.Redis(host=config.get('cache', 'host'),
                    port=config.get('cache', 'port'),
                    decode_responses=True)

# load files containing unique requests ids, mot ids of real trips, and conversion from mot text to id
base_dir = 'categories/loader'
filename_request_id = 'unique_request_id.txt'
filename_tripid_to_mot = 'tripid_to_motid.json'
filename_mot_text_to_id = 'mote_text_to_motid.json'
abs_file_requests_id = os.path.join(base_dir, filename_request_id)
abs_file_tripid_to_mot = os.path.join(base_dir, filename_tripid_to_mot)
abs_file_mot_text_to_id = os.path.join(base_dir, filename_mot_text_to_id)
unique_requests_id = open(abs_file_requests_id, 'r').read().split('\n')
tripid_to_motsid = json.load(open(abs_file_tripid_to_mot))
mot_text_to_id = json.load(open(abs_file_mot_text_to_id))

# average speeds (to compute lengths)
walking_avg_speed = 6 / 60  # in km/min
bike_avg_speed = 19 / 60  # in km/min

# create an empty dataframe to store the results
df = pd.DataFrame()
df_request_id = pd.DataFrame()

# create list to store requests whose solutions the user did not choose
request_id_no_solution = list()

k = 1
NPRINT = 100
for request_id in unique_requests_id[0:5000]:

    # request_id = '#30:10002'
    # user_id = cache.get('{request_id}:user_id'.format(request_id=request_id))

    output_request_level, \
    output_offer_level, \
    output_tripleg_level = read_data_from_cache_wrapper(pa_cache=cache, pa_request_id=request_id,
                                                        pa_request_level_items=['user_id', 'from_lat', 'from_lon',
                                                                                'to_lat', 'to_lon'],
                                                        pa_offer_level_items=['duration', 'weather', 'complete_total',
                                                                              'co2',
                                                                              'distance',
                                                                              'n_monuments'],
                                                        pa_tripleg_level_items=['privacy_level',
                                                                                'seating_quality',
                                                                                'transportation_mode',
                                                                                'leg_stops',
                                                                                'duration'])
    user_id = output_request_level['user_id']
    actual_modes_of_transport = tripid_to_motsid[request_id]

    # dictionaries to store results
    offer_durations_xsd = dict()
    offer_weather = dict()
    offer_co2 = dict()
    offer_distance_covered = dict()
    offer_num_stops = dict()
    offer_complete_total = dict()
    offer_privacy_level = dict()
    offer_seating_quality = dict()
    offer_n_monuments = dict()
    offer_social = dict()
    offer_fraction_bike_walk = dict()
    offer_length_bike_walking = dict()
    offer_response = dict()

    if 'offer_ids' in output_offer_level.keys():
        # offer-level information
        for offer_id in output_offer_level['offer_ids']:
            # dictionary of durations (quick category)
            offer_durations_xsd.setdefault(offer_id, output_offer_level[offer_id]['duration'])
            # dictionary of prob_delay because of weather (reliable category)
            offer_weather.setdefault(offer_id, float(output_offer_level[offer_id]['weather']))
            # dictionary of co2 emissions (environmentally friendly category)
            offer_co2.setdefault(offer_id, output_offer_level[offer_id]['co2'])
            # dictionary of total price (cheap category)
            offer_complete_total.setdefault(offer_id, output_offer_level[offer_id]['complete_total']['value'])
            # dictionary of distance covered (short category)
            offer_distance_covered.setdefault(offer_id, float(output_offer_level[offer_id]['distance']))
            # dictionary of number of stops (short category)
            offer_num_stops.setdefault(offer_id, len(output_tripleg_level[offer_id]['triplegs']) - 1)
            # dictionary of number of monuments
            offer_n_monuments.setdefault(offer_id, float(output_offer_level[offer_id]['n_monuments']))
            # list of privacy level (multitasking and comfortable category)
            offer_privacy_level_list = list()
            # list of seating quality (comfortable category)
            offer_seating_quality_list = list()
            # variables to store the number of legs done by bike/walk and distance
            legs_bike_walk = 0
            distance_bike_walk = 0.0
            # modes of transport (to find the actual response)
            modes_of_transport = list()

            # leg-level information
            if 'triplegs' in output_tripleg_level[offer_id].keys():

                # total number of legs
                n_legs = len(output_tripleg_level[offer_id]['triplegs'])

                # social: ridesharing?
                ridesharing = 0

                for leg_id in output_tripleg_level[offer_id]['triplegs']:
                    # privacy level
                    offer_privacy_level_list.append(float(output_tripleg_level[offer_id][leg_id]['privacy_level']))
                    # seating quality
                    offer_seating_quality_list.append(float(output_tripleg_level[offer_id][leg_id]['seating_quality']))
                    # social: ridesharing?
                    if output_tripleg_level[offer_id][leg_id]['transportation_mode'] in ['others-drive-car',
                                                                                         'bikesharing']:
                        ridesharing = 1

                    # legs done by bike/walk (healthy and d2d)
                    if output_tripleg_level[offer_id][leg_id]['transportation_mode'] in ['cycle', 'walk']:
                        legs_bike_walk += 1
                        if output_tripleg_level[offer_id][leg_id]['transportation_mode'] == 'walk':
                            distance_bike_walk += walking_avg_speed * compute_single_duration(
                                output_tripleg_level[offer_id][leg_id]['duration']
                            )

                        if output_tripleg_level[offer_id][leg_id]['transportation_mode'] == 'cycle':
                            distance_bike_walk += bike_avg_speed * compute_single_duration(
                                output_tripleg_level[offer_id][leg_id]['duration']
                            )

                    modes_of_transport.append(mot_text_to_id[
                                                  output_tripleg_level[offer_id][leg_id]['transportation_mode']])

                offer_privacy_level.setdefault(offer_id, np.mean(offer_privacy_level_list))
                offer_seating_quality.setdefault(offer_id, np.mean(offer_seating_quality_list))
                offer_social.setdefault(offer_id, ridesharing)
                offer_fraction_bike_walk.setdefault(offer_id,
                                                    legs_bike_walk / len(output_tripleg_level[offer_id]['triplegs']))
                # offer_length_bike_walking.setdefault(offer_id,
                #                                     distance_bike_walk / float(
                #                                         output_offer_level[offer_id]['distance']))
                offer_length_bike_walking.setdefault(offer_id, distance_bike_walk)

                # find the alternative the user chose
                offer_response.setdefault(offer_id, check_if_equal(modes_of_transport, actual_modes_of_transport))
    if np.sum(list(offer_response.values())) == 0:
        request_id_no_solution.append(request_id)
    # normalize each feature
    # compute normalized duration
    offer_durations_normalized = zscore(compute_duration(offer_durations_xsd), flipped=True)
    # compute normalized weather
    offer_weather_normalized = zscore(offer_weather, flipped=True)
    # compute normalized price
    offer_complete_total_normalized = compute_normalized_complete_total(offer_complete_total)
    # compute normalized distance covered
    offer_distance_covered_normalized = zscore(offer_distance_covered, flipped=True)
    # compute normalized number of stops
    offer_num_stops_normalized = zscore(offer_num_stops, flipped=True)
    # compute normalized walking/biking distance (d2d)
    offer_walking_distance_normalized = compute_normalized_bike_walk_dist(offer_length_bike_walking,
                                                                          offer_distance_covered, flipped=True)
    # compute normalized co2 per km
    offer_co2_km_normalized = compute_normalized_co2_km(offer_co2, offer_distance_covered)
    # compute normalized privacy level
    offer_privacy_level_normalized = zscore(offer_privacy_level)
    # compute normalized seating quality
    offer_seating_quality_normalized = zscore(offer_seating_quality)
    # compute normalized number of monuments
    offer_n_monuments_normalized = zscore(offer_n_monuments)
    # compute normalized fraction of trip done by walk/bike
    offer_fraction_bike_walk_normalized = zscore(offer_fraction_bike_walk)
    # compute normalized length fraction of trip done by walk/bike (healthy)
    offer_length_bike_walking_normalized = compute_normalized_bike_walk_dist(offer_length_bike_walking,
                                                                             offer_distance_covered)

    # compute scores of each category
    # comfortable: seating quality, privacy level and weather (in that order)
    comfortable_dict = dict([(k, [offer_seating_quality_normalized[k],
                                  offer_privacy_level_normalized[k],
                                  offer_weather_normalized[k]]) for k in offer_seating_quality_normalized])
    comfortable_scores = rod_aggregation(comfortable_dict)

    # short: distance covered, number of stops (in that order)
    short_dict = dict([(k, [offer_distance_covered_normalized[k],
                            offer_num_stops_normalized[k]]) for k in offer_distance_covered_normalized])
    short_scores = rod_aggregation(short_dict)

    # healthy: length fraction and leg fraction (in that order)
    healthy_dict = dict([(k, [offer_length_bike_walking_normalized[k],
                              offer_fraction_bike_walk_normalized[k]]) for k in offer_distance_covered_normalized])
    healthy_scores = rod_aggregation(healthy_dict)

    # store all scores in the same dictionary
    offer_scores = dict([(k, [user_id, offer_durations_normalized[k], offer_weather_normalized[k],
                              offer_complete_total_normalized[k], comfortable_scores[k],
                              offer_walking_distance_normalized[k], offer_co2_km_normalized[k],
                              short_scores[k], offer_privacy_level_normalized[k], offer_social[k],
                              offer_n_monuments_normalized[k], healthy_scores[k], offer_response[k]])
                         for k in offer_seating_quality_normalized])

    df = pd.concat([df, pd.DataFrame.from_dict(offer_scores, orient='index')])
    df_request_id = pd.concat([df_request_id, pd.DataFrame(len(offer_scores) * [request_id])])
    if k % NPRINT == 0:
        print('{k} trips categorized'.format(k=k))
    k += 1

df = pd.concat([df_request_id.reset_index().rename(columns={0: 'request_id'}),
                df.reset_index().rename(columns={'index': 'offer_id'})], axis=1)

# drop unnecessary columns
df = df.drop(columns=['index'])
# rename columns
df = df.rename(columns={0: 'user_id', 1: 'Quick', 2: 'Reliable', 3: 'Cheap',
                        4: 'Comfortable', 5: 'D2D', 6: 'Env_Friendly', 7: 'Short', 8: 'Multitasking',
                        9: 'Social', 10: 'Panoramic', 11: 'Healthy', 12: 'Response'})
print(df)
print(df.Response.sum())
# print(request_id_no_solution)
np.savetxt('categories/data/request_id_no_solution_5000.txt', request_id_no_solution, fmt='%s')
df.to_csv('categories/data/df_combined_5000.csv')
