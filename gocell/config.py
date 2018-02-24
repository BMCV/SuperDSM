import json


def pop_value(kwargs, kw, default):
    return kwargs.pop(kw) if kw in kwargs else default


def set_default_value(kwargs, kw, default):
    if kw not in kwargs: kwargs[kw] = default


def get_value(config, key, default):
    if key not in config: config[key] = default
    return config[key]


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

