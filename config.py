from typing import List, Optional
from icecream import ic

class ObjectDict(dict):
    def __init__(self, d: Optional[dict] = None):
        super().__init__()
        if d is not None:
            self.add_from_dict(d)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    @classmethod
    def parse_from_yml(cls, filename: str):
        import yaml

        with open(filename, "r") as f:
            args = yaml.safe_load(f)
        return cls(args)

    def add_from_dict(self, d: dict):
        r"""
        Merge from other dict-like object
        """
        for key, value in d.items():
            # ic(key, value)
            if isinstance(value, dict):
                value = ObjectDict(value)
            self[key] = value


# Alias for ObjectDict
# class Config(dict):
#     def __init__(self):
#         super().__init__()
Config = ObjectDict
