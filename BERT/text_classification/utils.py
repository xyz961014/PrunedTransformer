

def add_attr_from_dict(obj, dic):
    for key, value in dic.items():
        setattr(obj, key, value)
    return obj
