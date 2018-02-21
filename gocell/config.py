def pop_value(kwargs, kw, default):
    return kwargs.pop(kw) if kw in kwargs else default


def set_default_value(kwargs, kw, default):
    if kw not in kwargs: kwargs[kw] = default


def get_value(config, key, default):
    if key not in config: config[key] = default
    return config[key]

