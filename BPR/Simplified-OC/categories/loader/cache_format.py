from datetime import datetime, timedelta
from geojson import LineString
from random import randint
import numpy as np
import json
import os


def random_with_n_digits(n):
    """
    This function generates a random number of 'n' digits. Used to assign a unique id to each offer
    """
    range_start = 10 ** (n - 1)
    range_end = (10 ** n) - 1
    return randint(range_start, range_end)


def leg_type(mode):
    """
    This function classifies the leg depending on the mode of transport (continuous, timed and ridesharing)
    """
    continuous_leg = ['walking', 'bike', 'car']
    timed_leg = ['train', 'taxi', 'change', 'bus', 'subway', 'tram', 'genericpubtrans', 'boat', 'funicular']
    ridesharing_leg = ['carsharing', 'bikesharing']
    if mode in continuous_leg:
        return 'continuous'
    if mode in timed_leg:
        return 'timed'
    if mode in ridesharing_leg:
        return 'ridesharing'


def mapping_transport_mode(t_mode):
    """
    This function maps the mode of transport to the names specify in the offer cache schema (it solves the
    discrepancies between the routeRANK and TRIAS format)
    """
    transports_to_map = {'bike': 'cycle',
                         'boat': 'water',
                         'car': 'self-drive-car',
                         'carsharing': 'others-drive-car',
                         'walking': 'walk',
                         'train': 'rail',
                         'subway': 'metro'}
    if t_mode in transports_to_map:
        return transports_to_map[t_mode]
    else:
        return t_mode


def mapping_privacy_level(t_mode):
    """
    This function assigns a predefined privacy level depending on the transport mode
    """
    privacy_dict = {'cycle': 5,
                    'water': 3,
                    'self-drive-car': 5,
                    'others-drive-car': 1,
                    'walk': 5,
                    'rail': 2,
                    'metro': 2,
                    'taxi': 1,
                    'change': 3,
                    'bus': 2,
                    'tram': 2,
                    'bikesharing': 1,
                    'funicular': 2,
                    'genericpubtrans': 2}

    return privacy_dict[t_mode]


def mapping_seat_quality(t_mode):
    """
    This function assigns a predefined seating quality depending on the transport mode
    """
    comfort_dict = {'cycle': 2,
                    'water': 3,
                    'self-drive-car': 5,
                    'others-drive-car': 5,
                    'walk': 0,
                    'rail': 4,
                    'metro': 3,
                    'taxi': 5,
                    'change': 3,
                    'bus': 4,
                    'tram': 3,
                    'bikesharing': 2,
                    'funicular': 3,
                    'genericpubtrans': 3}
    return comfort_dict[t_mode]


def get_coordinates(offer, place):
    """
    This functions gets the coordinates from a place of a given routeRANK offer
    """
    lat, lon = offer['places'][place]['latitude'], offer['places'][place]['longitude']
    return float(lat), float(lon)


def get_time_format(dur):
    """
    This function writes the duration in the format specified by the offer cache schema
    """
    sec = timedelta(seconds=dur)
    d = datetime(10, 10, 10) + sec
    return 'P%dY%dM%dDT%dH%dM%dS' % (d.year - 10, d.month - 10, d.day - 10, d.hour, d.minute, d.second)


base_dir = 'categories/loader'
filename_delay = 'delay_tripid.json'
filename_user = 'tripid_to_userid.json'
abs_file_delay = os.path.join(base_dir, filename_delay)
abs_file_user = os.path.join(base_dir, filename_user)
tripid_to_delay = json.load(open(abs_file_delay))
tripid_to_userid = json.load(open(abs_file_user))


def get_weather(tripid):
    prob_delay = tripid_to_delay.get(tripid, 0)
    return prob_delay


def transform_trip(trip):
    request_id = '{tripId}'.format(tripId=trip['tripId'])
    request_new_format = dict()
    request_new_format.setdefault(request_id, {})
    # get user_id
    request_new_format[request_id].setdefault('user_id', tripid_to_userid[request_id])
    # get starting and ending location from trip request
    request_new_format[request_id].setdefault('from', trip['from'])
    request_new_format[request_id].setdefault('to', trip['to'])

    fmt = '%H:%M'
    offer_list = []
    for alternative in trip['alternatives']:
        # offer level information
        # get the offer ID and store it (generated randomly)
        offer_id = str(random_with_n_digits(16))
        offer_list.append(offer_id)
        request_new_format[request_id].setdefault(offer_id, {})

        # get the starting time and store it
        start_time = alternative[0]['segments'][0]['departureTime']
        start_date = alternative[0]['segments'][0]['departureDate']
        t = '{start_date} {start_time}'.format(start_date=start_date, start_time=start_time)
        starttime = datetime.fromisoformat(t).isoformat()
        request_new_format[request_id][offer_id].setdefault('start_time', starttime)

        # get the ending time and store it
        end_time = alternative[-1]['segments'][-1]['arrivalTime']
        end_date = alternative[-1]['segments'][-1]['arrivalDate']
        t = '{end_date} {end_time}'.format(end_date=end_date, end_time=end_time)
        endtime = datetime.fromisoformat(t).isoformat()
        request_new_format[request_id][offer_id].setdefault('end_time', endtime)

        # get duration and store it
        # tdelta = datetime.strptime(end_time, fmt) - datetime.strptime(start_time, fmt)
        # request_new_format[request_id][offer_id].setdefault('duration', get_time_format(tdelta.total_seconds()))

        # to compute the total price of an offer
        price = 0.0
        co2 = 0.0
        distance = 0.0

        # count to store the leg_id
        i = 0
        leg_ids = []
        accumulated_seconds = timedelta(seconds=0)
        for segments in alternative:
            price += segments['totals']['price']
            co2 += segments['totals']['co2']
            for segment in segments['segments']:
                distance += segment['distance']
                for leg in segment['legs']:
                    # storing the id for each leg
                    leg_id = '{i}-{offerid}'.format(i=i, offerid=offer_id)
                    leg_ids.append(leg_id)
                    i += 1

                    # leg type
                    request_new_format[request_id][offer_id].setdefault(leg_id, {})
                    legtype = leg_type(leg['transport'])
                    request_new_format[request_id][offer_id][leg_id].setdefault('leg_type', legtype)

                    # departure time
                    start_time = leg['departureTime']
                    start_date = leg['departureDate']
                    t = '{start_date} {start_time}'.format(start_date=start_date, start_time=start_time)
                    starttime = datetime.fromisoformat(t)
                    request_new_format[request_id][offer_id][leg_id].setdefault('start_time', starttime.isoformat())

                    # arrival time
                    end_time = leg['arrivalTime']
                    end_date = leg['arrivalDate']
                    t = '{end_date} {end_time}'.format(end_date=end_date, end_time=end_time)
                    endtime = datetime.fromisoformat(t)
                    request_new_format[request_id][offer_id][leg_id].setdefault('end_time', endtime.isoformat())

                    # duration
                    tdelta = datetime.strptime(end_time, fmt) - datetime.strptime(start_time, fmt)
                    accumulated_seconds += endtime - starttime
                    request_new_format[request_id][offer_id][leg_id].setdefault('duration',
                                                                                get_time_format(tdelta.total_seconds()))

                    # transportation mode
                    request_new_format[request_id][offer_id][leg_id].setdefault('transportation_mode',
                                                                                mapping_transport_mode(
                                                                                    leg['transport']))

                    # leg stops
                    coord_ini = get_coordinates(trip, leg['from'])
                    coord_fin = get_coordinates(trip, leg['to'])
                    # example_new_format[request_id][offer_id][leg_id].setdefault('leg_stops',[coord_ini,coord_fin])
                    request_new_format[request_id][offer_id][leg_id].setdefault('leg_stops',
                                                                                LineString([coord_ini, coord_fin]))

                    # privacy_level
                    request_new_format[request_id][offer_id][leg_id].setdefault('privacy_level',
                                                                                mapping_privacy_level(
                                                                                    mapping_transport_mode(
                                                                                        leg['transport'])))

                    # seating quality
                    request_new_format[request_id][offer_id][leg_id].setdefault('seating_quality',
                                                                                mapping_seat_quality(
                                                                                    mapping_transport_mode(
                                                                                        leg['transport'])))

        request_new_format[request_id][offer_id].setdefault('duration',
                                                            get_time_format(accumulated_seconds.total_seconds()))
        request_new_format[request_id][offer_id].setdefault('num_interchanges', len(leg_ids) - 1)
        price_dict = {'value': int(price * 100), 'currency': 'EUR'}
        request_new_format[request_id][offer_id].setdefault('complete_total', price_dict)
        request_new_format[request_id][offer_id].setdefault('co2', co2)
        request_new_format[request_id][offer_id].setdefault('distance', distance)
        request_new_format[request_id][offer_id].setdefault('n_monuments', np.random.randint(0, 5))
        request_new_format[request_id][offer_id].setdefault('legs', leg_ids)
        request_new_format[request_id][offer_id].setdefault('weather', get_weather(request_id))
    request_new_format[request_id].setdefault('offers', offer_list)
    # for segments in alternative:
    return request_new_format
