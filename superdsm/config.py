import json
import hashlib


def _cleanup_value(value):
    return value.entries if isinstance(value, Config) else value


class Config:
    """Represents a set of hyperparameters.
    """

    def __init__(self, other=None):
        if isinstance(other, dict):
            self.entries = other
        elif isinstance(other, Config):
            self.entries = json.loads(json.dumps(other.entries))
        else:
            raise ValueError(f'Unknown argument: ' + other)

    def pop(self, key, default):
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.pop(keys[-1], default)
        else:
            return self.entries.pop(key, default)

    def set_default(self, key, default, override_none=False):
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.set_default(key, {}, override_none)
            return config.set_default(keys[-1], default, override_none)
        else:
            if key not in self.entries or (override_none and self.entries[key] is None):
                self.entries[key] = _cleanup_value(default)
            return self[key]

    def get(self, key, default):
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.get(keys[-1], default)
        else:
            if key not in self.entries: self.entries[key] = _cleanup_value(default)
            value = self.entries[key]
            return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key):
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config[key]
            return config[keys[-1]]
        else:
            value = self.entries[key]
            return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def update(self, key, func):
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.update(keys[-1], func)
        else:
            self.entries[key] = _cleanup_value(func(self.entries.get(key, None)))
            return self.entries[key]

    def __setitem__(self, key, value):
        self.update(key, lambda *args: value)
        return self

    def merge(self, config_override):
        for key, val in _cleanup_value(config_override).items():
            if not isinstance(val, dict):
                self.entries[key] = val
            else:
                self.get(key, {}).merge(val)
        return self

    def copy(self):
        return Config(self)

    def derive(self, config_override):
        return self.copy().merge(config_override)

    def dump_json(self, fp):
        json.dump(self.entries, fp)
        
    @property
    def md5(self):
        return hashlib.md5(json.dumps(self.entries).encode('utf8'))

