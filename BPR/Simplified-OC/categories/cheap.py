from categories.utils.utils import zscore


def compute_normalized_complete_total(complete_total_dict):
    """
    This function returns the normalized price of each alternative
    - Input: dictionary containing the total price of each one of the alternatives
    in the format specified by the offer-cache
    """
    complete_total_euros = dict()
    for key, price in complete_total_dict.items():
        price_euros = float(price)/100
        complete_total_euros.setdefault(key, price_euros)
    # normalize values
    price_normalized = zscore(complete_total_euros, flipped=True)
    return price_normalized
