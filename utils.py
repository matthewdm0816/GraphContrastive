r"""
General Helpers/Utilities
"""
from icecream import ic
from tqdm import tqdm, trange
from contextlib import contextmanager
import colorama, os, pretty_errors
import torch
import torch.nn as nn

colorama.init(autoreset=True)

from yaml import load, dump, safe_load

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
    r"""
    Wrap the dict in an object-like way
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


class YAMLParser:
    r"""
    Simply parses YAML file and return an object-like dictionary
    """

    def __init__(self, filename):
        with open(filename, "r") as f:
            self.data = safe_load(f)
        self.data = ObjectDict(self.data)

    @contextmanager
    def fetch(self):
        yield self.data


def parallelize_model(model, device, gpu_ids, gpu_id):
    from torch_geometric.nn import DataParallel

    try:
        # fix sync-batchnorm
        from sync_batchnorm import convert_model

        model = convert_model(model)
    except ModuleNotFoundError:
        raise ModuleNotFoundError("Sync-BN plugin not found")
    # NOTE: DataParallel call MUST after model definition completes
    model = DataParallel(model, device_ids=gpu_ids, output_device=gpu_id).to(device)
    ic("Parallelized model")
    return model


def init_train(parallel, gpu_ids):
    torch.backends.cudnn.benchmark = True
    print(
        colorama.Fore.MAGENTA
        + (
            "Running in Single-GPU mode"
            if not parallel
            else "Running in Multiple-GPU mode with GPU {}".format(gpu_ids)
        )
    )

    # load timestamp
    try:
        with open("timestamp.yml", "r") as f:
            timestamp = safe_load(f)["timestamp"] + 1
    except FileNotFoundError:
        # init timestamp
        timestamp = 1
    finally:
        # save timestamp
        with open("timestamp.json", "w") as f:
            dump({"timestamp": timestamp}, f)
    return timestamp


def get_optimizer(model, optimizer_type, lr, beg_epochs, T_0=200, T_mult=1):
    from torch import optim
    import re

    print(colorama.Fore.RED + "Using optimizer type %s" % optimizer_type)
    if optimizer_type == "Adam":
        optimizer = optim.Adam(
            [
                {"params": model.parameters(), "initial_lr": lr},
                # {"params": model.parameters(), "initial_lr": 0.002}
            ],
            lr=lr,
            weight_decay=5e-4,
            betas=(0.9, 0.999),
        )
    elif optimizer_type == "SGD":
        # Using SGD Nesterov-accelerated with Momentum
        optimizer = optim.SGD(
            [{"params": model.parameters(), "initial_lr": lr,},],
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
            nesterov=True,
        )

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=100, last_epoch=beg_epochs
    # )
    # Cosine annealing with restarts
    # etc.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, last_epoch=beg_epochs, eta_min=1e-6
    )
    return optimizer, scheduler

def load_model(model, optimizer, path: str, e: int, evaluate=None, postfix: str='latest'):
    model_path = os.path.join(path, 'model-%s.save' % postfix)
    optimizer_path = os.path.join(path, 'opt-%s.save' % postfix)
    loaded = torch.load(model_path)
    try:
        loaded = loaded.module
        ic(loaded)
    except:
        ic()
        pass
    model.load_state_dict(loaded)
    print("Loaded milestone with epoch %d at %s" % (e, model_path))
    optimizer.load_state_dict(torch.load(optimizer_path))
    print("Loaded milestone optimizer with epoch %d at %s" % (e, optimizer_path))
    # if evaluate is not None:
    #     evaluate(model)

def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
