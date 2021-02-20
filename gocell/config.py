import json


def pop_value(kwargs, kw, default):
    return kwargs.pop(kw) if kw in kwargs else default


def set_default_value(kwargs, kw, default):
    if '/' in kw:
        keys = kw.split('/')
        for key in keys[:-1]:
            kwargs = set_default_value(kwargs, key, {})
        return set_default_value(kwargs, keys[-1], default)
    else:
        if kw not in kwargs: kwargs[kw] = default
        return kwargs[kw]


def get_value(config, key, default):
    if '/' in key:
        keys = key.split('/')
        for key in keys[:-1]:
            config = get_value(config, key, {})
        return get_value(config, keys[-1], default)
    else:
        if key not in config: config[key] = default
        return config[key]


def update_value(config, key, func):
    if '/' in key:
        keys = key.split('/')
        for key in keys[:-1]:
            config = get_value(config, key, {})
        return update_value(config, keys[-1], func)
    else:
        config[key] = func(config[key])
        return config[key]


def set_value(config, key, value):
    update_value(config, key, lambda *args: value)


def update(base_cfg, cfg_override):
    for key, val in cfg_override.items():
        if key not in base_cfg.keys() or not isinstance(val, dict):
            base_cfg[key] = val
        else:
            if not isinstance(base_cfg[key], dict):
                base_cfg[key] = {}
            update(base_cfg[key], val)
    return base_cfg


def derive(base_cfg, cfg_override):
    cfg = json.loads(json.dumps(base_cfg))
    return update(cfg, cfg_override)


def copy(base_cfg):
    return derive(base_cfg, dict())

