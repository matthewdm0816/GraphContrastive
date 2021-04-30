r"""
General Helpers/Utilities
"""
from yaml import load, dump
from icecream import ic
from tqdm import tqdm, trange 
import pretty_errors
from contextlib import contextmanager
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    ic()
    from yaml import Loader, Dumper

class layers:
    r"""
    Enumerate layer sizes
    """
    def __init__(self, ns):
        self.ns = ns
        self.iter1 = iter(ns)
        self.iter2 = iter(ns)  # iterator of latter element
        next(self.iter2)

    def __iter__(self):
        return self

    def __next__(self):
        return (next(self.iter1), next(self.iter2))


class ObjectDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

class YAMLParser:
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self._data = load(f, Loader=Loader)
        self.data = ObjectDict(self.data)

    @property
    def data(self):
        return self._data

    @contextmanager
    def fetch(self):
        yield self.data
