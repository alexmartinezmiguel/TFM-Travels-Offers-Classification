#!/usr/bin/env python3
##############################################################################
# This script loads the routeRANK data into the cache

import sys
import geojson
import json
import argparse
import redis
import os
import numpy as np
import random
from r2r_offer_utils.cli_utils import IntRange

from loader.cache_format import transform_trip
from loader.data_cleaning import filter_trips, combine_alternatives

NPRINT = 100
number_trips = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-H', '--host',
                        default='localhost',
                        help='Redis hostname [default: localhost].')
    parser.add_argument('-p', '--port',
                        default=6379,
                        type=IntRange(1, 65536),
                        help='Redis port [default: 6379].')

    args = parser.parse_args()

    print("Connecting to Redis instance on {host}:{port}..."
          .format(host=args.host, port=args.port),
          file=sys.stderr, flush=True)
    redis = redis.Redis(host=args.host, port=args.port)

    base_dir = 'categories/data'
    file_1 = 'final1.json'
    file_2 = 'final2.json'
    path_file_1 = os.path.join(base_dir, file_1)
    path_file_2 = os.path.join(base_dir, file_2)
    routerank_alternatives_1 = json.load(open(path_file_1))
    routerank_alternatives_2 = json.load(open(path_file_2))
    routerank_alternatives = routerank_alternatives_1 + routerank_alternatives_2
    leg_alternatives = filter_trips(routerank_alternatives)

    new_trips = []
    base_dir = 'categories/loader'
    filename_motiv_request_id = 'unique_request_id.txt'
    abs_file_requests_id = os.path.join(base_dir, filename_motiv_request_id)
    unique_requests_id = open(abs_file_requests_id, 'r').read().split('\n')
    k = 0
    for unique_id in unique_requests_id[0:5000]:
        legs = []
        for trip in leg_alternatives:
            if trip['tripId'] == unique_id:
                legs.append(trip)
        if len(legs) < 8:
            new_trips.append(combine_alternatives(legs))
        else:
            print(k)
        if k % 100 == 0:
            print('{k} trips combined'.format(k=k))
        k += 1
    print(len(new_trips))

    i = 1
    random.seed(0)
    np.random.seed(0)
    for trip in new_trips:
        # get mobility request data
        mreq = transform_trip(trip)

        # request-level information
        request_id = '{tripId}'.format(tripId=trip['tripId'])

        # insert user_id
        redis.set('{request_id}:user_id'.format(request_id=request_id),
                  mreq[request_id]['user_id'])

        # insert starting coordinates from the trip request
        redis.set('{request_id}:from_lat'.format(request_id=request_id),
                  mreq[request_id]['from']['latitude'])
        redis.set('{request_id}:from_lon'.format(request_id=request_id),
                  mreq[request_id]['from']['longitude'])

        # insert ending coordinates from the trip request
        redis.set('{request_id}:to_lat'.format(request_id=request_id),
                  mreq[request_id]['to']['latitude'])
        redis.set('{request_id}:to_lon'.format(request_id=request_id),
                  mreq[request_id]['to']['longitude'])

        # insert a list of offer
        redis.lpush('{request_id}:offers'.format(request_id=request_id),
                    *mreq[request_id]['offers']
                    )

        # offer-level information
        for offer_id in mreq[request_id]['offers']:
            prefix = ('{request_id}:{offer_id}'
                      .format(request_id=request_id,
                              offer_id=offer_id)
                      )

            # start_time
            redis.set('{p}:start_time'.format(p=prefix),
                      mreq[request_id][offer_id]['start_time']
                      )
            # end_time
            redis.set('{p}:end_time'.format(p=prefix),
                      mreq[request_id][offer_id]['end_time']
                      )
            # duration
            redis.set('{p}:duration'.format(p=prefix),
                      mreq[request_id][offer_id]['duration']
                      )
            # num_interchanges
            redis.set('{p}:num_interchanges'.format(p=prefix),
                      mreq[request_id][offer_id]['num_interchanges']
                      )
            # legs
            if mreq[request_id][offer_id]['legs']:
                redis.lpush('{p}:legs'.format(p=prefix),
                            *mreq[request_id][offer_id]['legs']
                            )
            # complete_total
            redis.hmset('{p}:complete_total'.format(p=prefix),
                        mreq[request_id][offer_id]['complete_total']
                        )
            # weather
            redis.set('{p}:weather'.format(p=prefix),
                      mreq[request_id][offer_id]['weather']
                      )
            # c02
            redis.set('{p}:co2'.format(p=prefix),
                      mreq[request_id][offer_id]['co2']
                      )
            # distance
            redis.set('{p}:distance'.format(p=prefix),
                      mreq[request_id][offer_id]['distance']
                      )
            # number monuments
            redis.set('{p}:n_monuments'.format(p=prefix),
                      mreq[request_id][offer_id]['n_monuments']
                      )
            # leg-level information
            for leg_id in mreq[request_id][offer_id]['legs']:
                prefix_leg = ('{request_id}:{offer_id}:{leg_id}'
                              .format(request_id=request_id,
                                      offer_id=offer_id,
                                      leg_id=leg_id)
                              )
                redis.set('{pl}:leg_type'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['leg_type']
                          )
                # start_time
                redis.set('{pl}:start_time'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['start_time']
                          )
                # end_time
                redis.set('{pl}:end_time'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['end_time']
                          )
                # duration
                redis.set('{pl}:duration'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['duration']
                          )
                # transportation_mode
                redis.set('{pl}:transportation_mode'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['transportation_mode']
                          )
                # leg_stops
                redis.set('{pl}:leg_stops'.format(pl=prefix_leg),
                          geojson.dumps(mreq[request_id][offer_id][leg_id]['leg_stops'])
                          )
                # seating_quality
                redis.set('{pl}:seating_quality'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['seating_quality']
                          )
                # privacy_level
                redis.set('{pl}:privacy_level'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['privacy_level']
                          )
                # seating_quality
                redis.set('{pl}:seating_quality'.format(pl=prefix_leg),
                          mreq[request_id][offer_id][leg_id]['seating_quality']
                          )
        if i % NPRINT == 0:
            print('insert #{} (request_id: {})'.format(i, request_id),
                  file=sys.stderr, flush=True)
        i += 1
    print('All data loaded!', file=sys.stderr)
