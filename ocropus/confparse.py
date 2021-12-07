import yaml


class Params:
    """Wrapper for dictionaries that allows attribute-style access."""

    def __init__(self, d, exclude=["__class__", "self"]):
        assert isinstance(d, dict)
        self.__the_dict__ = {k: v for k, v in d.items() if k not in exclude}

    def get(self, *args):
        return self.__the_dict__.get(*args)

    def __getitem__(self, name):
        if name not in self.__the_dict__:
            raise KeyError(name)
        return self.__the_dict__[name]

    def __setitem__(self, name, value):
        self.__the_dict__[name] = value

    def __getattr__(self, name):
        if name not in self.__the_dict__:
            raise AttributeError(name)
        value = self.__the_dict__[name]
        if isinstance(value, dict):
            return Params(value)
        else:
            return value

    def dict(self):
        return self.__the_dict__

    def __setattr__(self, name, value):
        if name[0] == "_":
            object.__setattr__(self, name, value)
        else:
            self.__the_dict__[name] = value

    def __getstate__(self):
        return self.__the_dict__

    def __setstate__(self, state):
        self.__the_dict__ = state


def update_config(config, updates, path=None):
    """Update the given config with the given updates."""
    path = path or []
    if isinstance(config, dict) and isinstance(updates, dict):
        for k, v in updates.items():
            if isinstance(config.get(k), dict):
                update_config(config.get(k), v, path=path + [k])
            else:
                config[k] = v
    else:
        raise ValueError(f"updates don't conform with config at {path}")


def scalar_convert(s):
    """Convert a scalar to an int/float if possible."""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s


def flatten_yaml(d, result=None, prefix=""):
    """Flatten a nested dictionary into a flat dictionary."""
    result = {} if result is None else result
    for k, v in d.items():
        if isinstance(v, dict):
            flatten_yaml(v, result=result, prefix=prefix + k + ".")
        else:
            result[prefix + k] = v
    return result


def set_config(config, key, value):
    """Set the given key in the given config to the given value."""
    path = key.split(".")
    for k in path[:-1]:
        config = config.setdefault(k, {})
    config[path[-1]] = scalar_convert(value)


def parse_args(argv, default_config):
    """Parse command line arguments.

    This starts with a default configuration and updates by merging with
    other .yaml configuration files and/or path.key=value assignments on
    the command line.
    """
    config = dict(default_config)
    if len(argv) < 1:
        return config
    for arg in argv:
        if "=" not in arg and (arg.endswith(".yaml") or arg.endswith(".yml")):
            arg = "config=" + arg
        if arg.startswith("config="):
            _, fname = arg.split("=", 1)
            with open(fname, "r") as stream:
                updates = yaml.safe_load(stream)
            update_config(config, updates)
            continue
        assert "=" in arg, arg
        key, value = arg.split("=", 1)
        set_config(config, key, value)
    return config
