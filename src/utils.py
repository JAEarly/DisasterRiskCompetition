
CLASS_ORDER = ['concrete_cement', 'healthy_metal', 'incomplete', 'irregular_metal', 'other']


class UnknownClassException(Exception):
    pass


def get_indexed_class_names():
    return [str(i) + "_" + class_name for (i, class_name) in enumerate(CLASS_ORDER)]


def get_indexed_class_name(class_name):
    if class_name in CLASS_ORDER:
        return str(CLASS_ORDER.index(class_name)) + "_" + class_name
    else:
        raise UnknownClassException(f'Unknown class "{class_name}"')
