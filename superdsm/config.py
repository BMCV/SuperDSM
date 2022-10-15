import json


def pop_config_value(kwargs, kw, default):
    return kwargs.pop(kw) if kw in kwargs else default


def set_config_default_value(kwargs, kw, default, override_none=False):
    if '/' in kw:
        keys = kw.split('/')
        for key in keys[:-1]:
            kwargs = set_config_default_value(kwargs, key, {}, override_none)
        return set_config_default_value(kwargs, keys[-1], default, override_none)
    else:
        if kw not in kwargs or (override_none and kwargs[kw] is None):
            kwargs[kw] = default
        return kwargs[kw]


def get_config_value(config, key, default):
    if '/' in key:
        keys = key.split('/')
        for key in keys[:-1]:
            config = get_config_value(config, key, {})
        return get_config_value(config, keys[-1], default)
    else:
        if key not in config: config[key] = default
        return config[key]


def update_config_value(config, key, func):
    if '/' in key:
        keys = key.split('/')
        for key in keys[:-1]:
            config = get_config_value(config, key, {})
        return update_config_value(config, keys[-1], func)
    else:
        config[key] = func(config[key])
        return config[key]


def set_config_value(config, key, value):
    update_config_value(config, key, lambda *args: value)


def update_config(base_cfg, cfg_override):
    for key, val in cfg_override.items():
        if key not in base_cfg.keys() or not isinstance(val, dict):
            base_cfg[key] = val
        else:
            if not isinstance(base_cfg[key], dict):
                base_cfg[key] = {}
            update_config(base_cfg[key], val)
    return base_cfg


def derive_config(base_cfg, cfg_override):
    cfg = json.loads(json.dumps(base_cfg))
    return update_config(cfg, cfg_override)


def copy_config(base_cfg):
    return derive_config(base_cfg, dict())

