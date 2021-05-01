import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary
import torch_geometric.nn as tgnn
from torch_geometric.nn import (
    GCNConv,
    SGConv,
    MessagePassing,
    knn_graph,
    DataParallel,
    GMMConv,
)
import torch_geometric as tg
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.utils import degree, get_laplacian, remove_self_loops
from torch_geometric.nn import GATConv, knn_graph
import torch_scatter as tscatter
from icecream import ic
from pprint import pprint

from tensorboardX import SummaryWriter
import time, random, os, sys, gc, copy, colorama, json, re, pretty_errors
from tabulate import tabulate
from gat_cls import DGCNNClassifier, GATClassifier
from utils import *

# limit CPU usage
torch.set_num_threads(16)
torch.set_num_interop_threads(16)
print(
    colorama.Fore.GREEN
    + "Using %d/%d cores/threads of CPU"
    % (torch.get_num_threads(), torch.get_num_interop_threads())
)

config_file = "config.yml"
config = YAMLParser(config_file).data

# ------------------ configuration tests ----------------- #

assert config.optimizer_type in ["Adam", "SGD"]
assert config.dataset_type in ["Cora", "Citeseer", "Pubmed"]
assert config.model_type in ["DGCNN", "GAT"]

# ---------------- general configurations ---------------- #

config.ngpu = len(config.gpu_ids)
config.parallel = config.ngpu > 1
config.batch_size = config.batch_size_single * config.ngpu
config.device = torch.device(
    "cuda:%d" % config.gpu_id if torch.cuda.is_available() else "cpu"
)

config.timestamp = init_train(config.parallel, config.gpu_ids)

# ---------------- dataset and dataloaders --------------- #

if config.dataset_type in ["Cora", "Citeseer", "Pubmed"]:
    config.dataset = Planetoid(
        root=os.path.join(config.dataset_path, config.dataset_type),
        name=config.dataset_type,
    )
    # add label noise
    # config.original_dataset = config.dataset
    if config.noise_rate > 1e-6:
        uniform_noise(config.dataset, config.noise_rate)
    config.train_dataset = config.dataset.data
    process_transductive_data(config.train_dataset, config.dataset.train_mask)
    config.val_dataset = config.dataset.data
    process_transductive_data(config.val_dataset, config.dataset.val_mask)
    config.test_dataset = config.dataset.data
    process_transductive_data(config.test_dataset, config.dataset.test_mask)
    if not config.parallel:
        config.train_loader = DataLoader(
            config.train_dataset, batch_size=config.batch_size
        )
        config.val_loader = DataLoader(config.val_dataset, batch_size=config.batch_size)
        config.test_loader = DataLoader(
            config.test_dataset, batch_size=config.batch_size
        )
    else:
        config.train_loader = DataListLoader(
            config.train_dataset, batch_size=config.batch_size
        )
        config.val_loader = DataListLoader(
            config.val_dataset, batch_size=config.batch_size
        )
        config.test_loader = DataListLoader(
            config.test_dataset, batch_size=config.batch_size
        )

else:
    raise NotImplementedError("Only supports Cora/Citeseer/Pubmed dataser for now")

config.batch_cnt = len(config.train_loader)

# ------------------ model configurations ---------------- #

config.fin = config.dataset.num_node_features
config.n_cls = config.dataset.num_classes
if config.model_type == "DGCNN":
    config.model = DGCNNClassifier(
        config.fin, config.n_cls, hidden_layers=config.hidden_layers
    )
elif config.model_type == "GAT":
    config.model = GATClassifier(
        config.fin, config.n_cls, hidden_layers=config.hidden_layers
    )
else:
    raise NotImplementedError


if config.parallel and config.use_sbn:
    config.model = parallelize_model(
        config.model, config.device, config.gpu_ids, config.gpu_id
    )
else:
    config.model = config.model.to(config.device)

config.writer = SummaryWriter(comment=config.model_name)

# --------------- optimizer configurations --------------- #

config.optimizer, config.scheduler = get_optimizer(
    config.model,
    config.optimizer_type,
    config.lr,
    config.beg_epochs,
    config.T_0,
    config.T_mult,
)

# ---------------- milestone optional load --------------- #

config.model_path = os.path.join(
    "model",
    "%s-%s-%s"
    % (
        config.model_name,
        str(config.timestamp),
        "clean" if config.noise_rate < 1e-6 else "ln%.1f" % config.noise_rate,
    ),
)
check_dir(config.model_path)

if config.milestone_path is not None:
    load_model(config.model, config.optimizer, config.milestone_path, config.beg_epochs)
else:
    init_weights(config.model)

# --------------------- print configs -------------------- #

print(tabulate(config.items()))
if config.debug:
    exit(0)

# ----------------- train/eval functions ----------------- #


def train(config, train: bool = True):
    r"""
    Train 1 epoch
    """
    if train:
        config.model.train()
    else:
        config.model.eval()

    # show current lr
    if train:
        current_lr = config.optimizer.param_groups[0]["lr"]
        ic(current_lr)
    total_loss, total_acc, total_oacc, total_confidence, total_entropy = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    with torch.set_grad_enabled(train):
        for i, batch in tqdm(
            enumerate(config.train_loader), total=config.train_loader_length
        ):
            # torch.cuda.empty_cache()
            batch = process_batch(batch, config.parallel, config.dataset_type)

            config.model.zero_grad()
            loss, x, conf, correct, original_correct, ent = config.model(batch)
            loss = loss.mean()
            acc = correct.sum() / x.shape[0]
            oacc = original_correct.sum() / x.shape[0]

            loss.backward()
            config.optimizer.step()

            if i % config.report_iterations == 0:
                print(
                    colorama.Fore.MAGENTA
                    + "[%d/%d]LOSS: %.2e, NACC: %.2f, CACC: %.2f, CONF: %.2f, ENT: %.2f"
                    % (
                        epoch,
                        i,
                        loss.detach().item(),
                        acc.detach().item(),
                        oacc.detach().item(),
                        conf.detach().item(),
                        ent.detach().item(),
                    )
                )
            with torch.no_grad():
                total_loss.add(loss)
                total_acc.add(acc)
                total_oacc.add(oacc)
                total_confidence.add(conf)
                total_entropy.add(ent)

    config.scheduler.step()
    if train:
        return (
            total_loss.mean(),
            total_acc.mean(),
            total_oacc.mean(),
            total_confidence.mean(),
            total_entropy.mean(),
            current_lr,
        )
    else:
        return (
            total_loss.mean(),
            total_acc.mean(),
            total_oacc.mean(),
            total_confidence.mean(),
            total_entropy.mean(),
        )


# ------------------------- main ------------------------- #

if __name__ == "__main__":
    for epoch in trange(config.beg_epochs, config.total_epochs + 1):
        (
            train_loss,
            train_acc,
            train_oacc,
            train_confidence,
            train_entropy,
            train_lr,
        ) = train(config, train=True)
        test_loss, test_acc, test_oacc, test_confidence, test_entropy = train(
            config, train=False
        )
        # ---------------------- save model ---------------------- #
        if epoch % config.milestone_save_epochs == 0 and epoch != 0:
            if config.parallel:
                model_to_save = config.model.module
            else:
                model_to_save = config.model
            torch.save(
                model_to_save.state_dict(),
                os.path.join(model_path, "model-%d.save" % (epoch)),
            )
            torch.save(
                config.optimizer.state_dict(),
                os.path.join(model_path, "opt-%d.save" % (epoch)),
            )
            torch.save(
                model_to_save.state_dict(),
                os.path.join(model_path, "model-latest.save"),
            )
            torch.save(
                config.optimizer.state_dict(),
                os.path.join(model_path, "opt-latest.save"),
            )

        # ------------------ log to tensorboard ------------------ #
        record_dict = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_oacc": train_oacc,
            "train_confidence": train_confidence,
            "train_entropy": train_entropy,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_oacc": test_oacc,
            "test_confidence": test_confidence,
            "test_entropy": test_entropy,
            "current_lr": current_lr,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                config.writer.add_scalar(key, record_dict[key], epoch)
            else:
                config.writer.add_scalars(key, record_dict[key], epoch)
                # add multiple records
