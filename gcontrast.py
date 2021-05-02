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
from gat_cls import *
from utils import *
from copy import copy

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
assert config.model_type in ["DGCNN", "GAT", "GCN"]

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
    config.fin = config.dataset.num_node_features
    config.n_cls = config.dataset.num_classes
    ic(config.dataset.data, config.fin, config.n_cls)
    ic(config.dataset.data.train_mask.sum().item())
    ic(config.dataset.data.val_mask.sum().item())
    ic(config.dataset.data.test_mask.sum().item())
    # add label noise
    # if config.noise_rate > 1e-6:
    uniform_noise(config.dataset.data, config.noise_rate)
    # ------------------- add feature noise ------------------ #

    if config.gaussian_noise_rate > 1e-6:
        add_l2_noise(config.dataset.data, config.gaussian_noise_rate)

    config.train_dataset = config.dataset.data.clone()
    process_transductive_data(config.train_dataset, config.dataset.data.train_mask)
    config.val_dataset = config.dataset.data.clone()
    process_transductive_data(config.val_dataset, config.dataset.data.val_mask)
    config.test_dataset = config.dataset.data.clone()
    process_transductive_data(config.test_dataset, config.dataset.data.test_mask)
    ic(config.train_dataset)
    # FIXME: single data dataset for now
    if not config.parallel:
        config.train_loader = DataLoader(
            [config.train_dataset], batch_size=config.batch_size
        )
        config.val_loader = DataLoader(
            [config.val_dataset], batch_size=config.batch_size
        )
        config.test_loader = DataLoader(
            [config.test_dataset], batch_size=config.batch_size
        )
    else:
        # todo
        config.train_loader = DataListLoader(
            [config.train_dataset], batch_size=config.batch_size
        )
        config.val_loader = DataListLoader(
            [config.val_dataset], batch_size=config.batch_size
        )
        config.test_loader = DataListLoader(
            [config.test_dataset], batch_size=config.batch_size
        )
else:
    raise NotImplementedError("Only supports Cora/Citeseer/Pubmed dataser for now")

config.batch_cnt = len(config.train_loader)

# ------------------ model configurations ---------------- #


if config.model_type == "DGCNN":
    config.model = DGCNNClassifier(
        config.fin,
        config.n_cls,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
elif config.model_type == "GAT":
    config.model = GATClassifier(
        config.fin,
        config.n_cls,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
    )
elif config.model_type == "GCN":
    config.model = GCNClassifier(
        config.fin,
        config.n_cls,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout,
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
    "%s-%s-%s-%s-%s"
    % (
        config.model_name,
        config.dataset_type,
        (config.adversarial_method + "%.3f" % config.adversarial_noise_rate)
        if config.adversarial_noise_rate > 1e-5
        else "noadv",
        "clean" if config.noise_rate < 1e-6 else "ln%.1f" % config.noise_rate,
        str(config.timestamp),
    ),
)
check_dir(config.model_path)

if config.milestone_path is not None:
    load_model(config.model, config.optimizer, config.milestone_path, config.beg_epochs)
else:
    init_weights(config.model)

# --------------------- print configs -------------------- #

config_str = tabulate(config.items())
print(config_str)
with open(config.tabulate_path, "w") as f:
    f.write(config_str)
if config.debug:
    exit(0)

# ----------------- train/eval functions ----------------- #


def train(config, current_epoch: int, loader, train: bool = True):
    r"""
    Train 1 epoch
    """
    if train:
        config.model.train()
        config.model.zero_grad()
    else:
        config.model.eval()

    # forward pass metrics
    total_loss, total_acc, total_oacc, total_confidence, total_entropy = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    # adversarial pass metrics
    (
        adv_total_loss,
        adv_total_acc,
        adv_total_oacc,
        adv_total_confidence,
        adv_total_entropy,
    ) = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    with torch.set_grad_enabled(train):
        # for i, batch in tqdm(enumerate(loader), total=len(loader)):
        for i, batch in enumerate(loader):
            # torch.cuda.empty_cache()
            batch = process_batch(
                batch, config.device, config.parallel, config.dataset_type
            )
            config.model.zero_grad()
            data_size = batch.x[batch.mask].shape[0]
            # forward pass
            loss, x, conf, correct, original_correct, ent = config.model(batch)
            loss = loss.mean()
            conf = conf.mean()
            ent = ent.mean()
            acc = correct.sum() / data_size
            oacc = original_correct.sum() / data_size
            # ic(data_size, correct.sum(), original_correct.sum())
            if train:
                # backward pass
                loss.backward()
                config.optimizer.step()
                config.scheduler.step()
                current_lr = config.optimizer.param_groups[0]["lr"]
            config.model.zero_grad()
            if current_epoch % config.report_iterations == 0:
                if train:
                    ic(current_lr)
                print(
                    colorama.Fore.MAGENTA
                    + "[%d%s/%d]LOSS: %.2e, NACC: %.2f%%, CACC: %.2f%%, CONF: %.2f, ENT: %.2f"
                    % (
                        epoch,
                        "t" if train else "e",
                        i,
                        loss.detach().item(),
                        100.0 * acc.detach().item(),
                        100.0 * oacc.detach().item(),
                        conf.detach().item(),
                        ent.detach().item(),
                    )
                )

            # adversarial pass, grad required
            with torch.set_grad_enabled(True):
                if config.adversarial_noise_rate > 1e-5:
                    if config.adversarial_method == "FGSM":
                        adv_batch = copy_batch(batch)
                        adv_batch.x.requires_grad = True
                        loss, _, _, _, _, _ = config.model(adv_batch)
                        loss = loss.mean()
                        loss.backward()
                        adv = adv_batch.x
                        norm = adv.abs().max(dim=-1)[0].mean()  # mean max
                        ic(norm, adv.abs().max())
                        vadv = (
                            torch.sgn(adv.grad) * norm * config.adversarial_noise_rate
                        )  # adv batch
                        adv = vadv + adv
                        # ic(adv_batch.y, adv_batch.y.shape)
                        adv_batch.x = adv
                        (
                            adv_loss,
                            _,
                            adv_conf,
                            adv_correct,
                            adv_original_correct,
                            adv_ent,
                        ) = config.model(adv_batch)
                        adv_loss = adv_loss.mean()
                        adv_conf = adv_conf.mean()
                        adv_ent = adv_ent.mean()
                        adv_acc = adv_correct.sum() / data_size
                        adv_oacc = adv_original_correct.sum() / data_size
                    else:
                        raise NotImplementedError
                else:
                    # fill stats with 0.
                    adv_loss, adv_conf, adv_acc, adv_oacc, adv_ent = torch.zeros([5])
                # record these adversarial metrics
                with torch.no_grad():
                    adv_total_loss.add(adv_loss)
                    adv_total_acc.add(adv_acc)
                    adv_total_oacc.add(adv_oacc)
                    adv_total_confidence.add(adv_conf)
                    adv_total_entropy.add(adv_ent)

                if current_epoch % config.report_iterations == 0:
                    print(
                        colorama.Fore.MAGENTA
                        + "[%d%s/%d]AdvLOSS: %.2e, AdvNACC: %.2f%%, AdvCACC: %.2f%%, AdvCONF: %.2f, AdvENT: %.2f"
                        % (
                            epoch,
                            "t" if train else "e",
                            i,
                            adv_loss.detach().item(),
                            100.0 * adv_acc.detach().item(),
                            100.0 * adv_oacc.detach().item(),
                            adv_conf.detach().item(),
                            adv_ent.detach().item(),
                        )
                    )

            with torch.no_grad():
                total_loss.add(loss)
                total_acc.add(acc)
                total_oacc.add(oacc)
                total_confidence.add(conf)
                total_entropy.add(ent)
    return (
        total_loss.mean,
        total_acc.mean,
        total_oacc.mean,
        total_confidence.mean,
        total_entropy.mean,
        adv_total_loss.mean,
        adv_total_acc.mean,
        adv_total_oacc.mean,
        adv_total_confidence.mean,
        adv_total_entropy.mean,
        None if not train else current_lr,
    )


# ------------------------- main ------------------------- #

if __name__ == "__main__":
    for epoch in range(config.beg_epochs, config.total_epochs + 1):
        (
            train_loss,
            train_acc,
            train_oacc,
            train_confidence,
            train_entropy,
            adv_train_loss,
            adv_train_acc,
            adv_train_oacc,
            adv_train_confidence,
            adv_train_entropy,
            current_lr,
        ) = train(config, epoch, loader=config.train_loader, train=True)
        (
            test_loss,
            test_acc,
            test_oacc,
            test_confidence,
            test_entropy,
            adv_test_loss,
            adv_test_acc,
            adv_test_oacc,
            adv_test_confidence,
            adv_test_entropy,
            _,
        ) = train(config, epoch, loader=config.test_loader, train=False)
        # ---------------------- save model ---------------------- #
        if epoch % config.milestone_save_epochs == 0 and epoch != 0:
            if config.parallel:
                model_to_save = config.model.module
            else:
                model_to_save = config.model
            torch.save(
                model_to_save.state_dict(),
                os.path.join(config.model_path, "model-%d.save" % (epoch)),
            )
            torch.save(
                config.optimizer.state_dict(),
                os.path.join(config.model_path, "opt-%d.save" % (epoch)),
            )
            torch.save(
                model_to_save.state_dict(),
                os.path.join(config.model_path, "model-latest.save"),
            )
            torch.save(
                config.optimizer.state_dict(),
                os.path.join(config.model_path, "opt-latest.save"),
            )

        # ------------------ log to tensorboard ------------------ #
        record_dict = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_oacc": train_oacc,
            "train_confidence": train_confidence,
            "train_entropy": train_entropy,
            "adv_train_loss": adv_train_loss,
            "adv_train_acc": adv_train_acc,
            "adv_train_oacc": adv_train_oacc,
            "adv_train_confidence": adv_train_confidence,
            "adv_train_entropy": adv_train_entropy,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_oacc": test_oacc,
            "test_confidence": test_confidence,
            "test_entropy": test_entropy,
            "adv_test_loss": adv_test_loss,
            "adv_test_acc": adv_test_acc,
            "adv_test_oacc": adv_test_oacc,
            "adv_test_confidence": adv_test_confidence,
            "adv_test_entropy": adv_test_entropy,
            "current_lr": current_lr,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                config.writer.add_scalar(key, record_dict[key], epoch)
            else:
                config.writer.add_scalars(key, record_dict[key], epoch)
                # add multiple records
