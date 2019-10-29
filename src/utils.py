import datetime


CLASSES = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']
LOCATIONS = {'colombia': ['borde_rural', 'borde_soacha'],
             'guatemala': ['mixco_1_and_ebenezer', 'mixco_3'],
             'st_lucia': ['castries', 'dennery', 'gros_islet']}


class UnknownClassException(Exception):
    pass


def get_indexed_class_names():
    return [str(i) + "_" + class_name for (i, class_name) in enumerate(CLASSES)]


def get_indexed_class_name(class_name):
    if class_name in CLASSES:
        return str(CLASSES.index(class_name)) + "_" + class_name
    else:
        raise UnknownClassException(f'Unknown class "{class_name}"')


def create_timestamp_str():
    today = datetime.datetime.now()
    return today.strftime("%Y-%m-%d_%X")
