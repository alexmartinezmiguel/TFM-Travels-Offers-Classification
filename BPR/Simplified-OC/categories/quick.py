import isodate


def compute_duration(durations):
    """
    This function returns the normalized duration of each alternative
    - Input: dictionary containing the duration of each one of the alternatives
    in the format specified by the offer-cache"""
    durations_minutes = dict()
    days_to_minute = 1440
    seconds_to_minutes = 1 / 60
    for key, duration in durations.items():
        # compute duration in minutes
        try:
            timedelta = isodate.parse_duration(duration)
            minutes = timedelta.days * days_to_minute + timedelta.seconds * seconds_to_minutes
            durations_minutes.setdefault(key, minutes)
        except isodate.isoerror.ISO8601Error:
            durations_minutes = 0.0
    return durations_minutes


def compute_single_duration(duration):
    """
    - Input: string with the duration in xsd format
    """
    days_to_minute = 1440
    seconds_to_minutes = 1 / 60
    try:
        timedelta = isodate.parse_duration(duration)
        minutes = timedelta.days * days_to_minute + timedelta.seconds * seconds_to_minutes
    except isodate.isoerror.ISO8601Error:
        minutes = 0.0
    return minutes
