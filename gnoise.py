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
from config import Config
from argparse import ArgumentParser
from gcnii import GCNIIClassifier

parser = ArgumentParser()
parser.add_argument("--config_files", type=str, nargs='+')
opt = parser.parse_args()

# Load Config Files
__C = Config()
for filename in opt.config_files:
    ic(filename)
    __C.add_from_dict(Config.parse_from_yml(filename))
    # ic(__C)
# ic(__C)

# Limit CPU usage
torch.set_num_threads(__C.CPU_THREADS)
torch.set_num_interop_threads(__C.CPU_THREADS)
print(
    colorama.Fore.GREEN
    + "Using %d/%d cores/threads of CPU"
    % (torch.get_num_threads(), torch.get_num_interop_threads())
)

# config_file = "__C.YML"
# config = YAMLParser(config_file).data

# ------------------ configuration tests ----------------- #

assert __C.OPTIMIZER_TYPE in ["Adam", "SGD", "AdamW"]
assert __C.DATASET_TYPE in ["Cora", "Citeseer", "Pubmed"]
assert __C.MODEL_TYPE in ["DGCNN", "GCN", "SGC", "GCNII"]

# ---------------- general configurations ---------------- #

__C.NGPU = len(__C.GPU_IDS)
__C.PARALLEL = __C.NGPU > 1
__C.BATCH_SIZE = __C.BATCH_SIZE_SINGLE * __C.NGPU
__C.DEVICE = torch.device(
    "cuda:%d" % __C.GPU_ID if torch.cuda.is_available() else "cpu"
)

__C.TIMESTAMP = init_train(__C.PARALLEL, __C.GPU_IDS)

# ---------------- dataset and dataloaders --------------- #

if __C.DATASET_TYPE in ["Cora", "Citeseer", "Pubmed"]:
    __C.DATASET = Planetoid(
        root=os.path.join(__C.DATASET_PATH, __C.DATASET_TYPE), name=__C.DATASET_TYPE,
    )
    __C.FIN = __C.DATASET.num_node_features
    __C.N_CLS = __C.DATASET.num_classes
    ic(__C.DATASET.data, __C.FIN, __C.N_CLS)
    ic(__C.DATASET.data.train_mask.sum().item())
    ic(__C.DATASET.data.val_mask.sum().item())
    ic(__C.DATASET.data.test_mask.sum().item())

    # ---------------- add label/feature noise --------------- #
    __C.LN_MASK = uniform_noise(__C.DATASET.data, __C.NOISE_RATE)  # NOTE: 1 if is noisy
    ic(__C.LN_MASK)
    gaussian_feature_noise(__C.DATASET.data, __C.GAUSSIAN_NOISE_RATE)

    __C.TRAIN_DATASET = __C.DATASET.data.clone()
    process_transductive_data(
        __C.TRAIN_DATASET, __C.DATASET.data.train_mask, __C.LN_MASK
    )
    __C.VAL_DATASET = __C.DATASET.data.clone()
    process_transductive_data(__C.VAL_DATASET, __C.DATASET.data.val_mask, __C.LN_MASK)
    __C.TEST_DATASET = __C.DATASET.data.clone()
    process_transductive_data(__C.TEST_DATASET, __C.DATASET.data.test_mask, __C.LN_MASK)
    ic(__C.TRAIN_DATASET)
    # FIXME: single data dataset for now
    if not __C.PARALLEL:
        __C.TRAIN_LOADER = DataLoader([__C.TRAIN_DATASET], batch_size=__C.BATCH_SIZE)
        __C.VAL_loader = DataLoader([__C.VAL_DATASET], batch_size=__C.BATCH_SIZE)
        __C.TEST_loader = DataLoader([__C.TEST_DATASET], batch_size=__C.BATCH_SIZE)
    else:
        # todo
        __C.TRAIN_LOADER = DataListLoader(
            [__C.TRAIN_DATASET], batch_size=__C.BATCH_SIZE
        )
        __C.VAL_loader = DataListLoader([__C.VAL_DATASET], batch_size=__C.BATCH_SIZE)
        __C.TEST_loader = DataListLoader([__C.TEST_DATASET], batch_size=__C.BATCH_SIZE)
else:
    raise NotImplementedError("Only supports Cora/Citeseer/Pubmed dataser for now")

__C.BATCH_CNT = len(__C.TRAIN_LOADER)

# ------------------ model configurations ---------------- #


if __C.MODEL_TYPE == "DGCNN":
    __C.MODEL = DGCNNClassifier(
        __C.FIN,
        __C.N_CLS,
        hidden_layers=__C.HIDDEN_LAYERS,
        dropout=__C.DROPOUT,
        knn=__C.KNN,
        khop=__C.KHOP,
    )
elif __C.MODEL_TYPE == "GCN":
    __C.MODEL = GCNClassifier(
        __C.FIN,
        __C.N_CLS,
        hidden_layers=__C.HIDDEN_LAYERS,
        dropout=__C.DROPOUT,
        knn=__C.KNN,
        khop=__C.KHOP,
    )
elif __C.MODEL_TYPE == "SGC":
    __C.MODEL = SGCClassifier(
        __C.FIN,
        __C.N_CLS,
        hidden_layers=__C.HIDDEN_LAYERS,
        dropout=__C.DROPOUT,
        knn=__C.KNN,
        khop=__C.KHOP,
    )
elif __C.MODEL_TYPE == "GCNII":
    __C.MODEL = GCNIIClassifier(__C.FIN, __C.N_CLS, __C)
else:
    raise NotImplementedError


if __C.PARALLEL and __C.USE_SBN:
    __C.MODEL = parallelize_model(__C.MODEL, __C.DEVICE, __C.GPU_IDS, __C.GPU_ID)
else:
    __C.MODEL = __C.MODEL.to(__C.DEVICE)

__C.MODEL_NAME = "%s-%s-%s-%s-%s-%s" % (
    __C.MODEL_NAME,
    __C.DATASET_TYPE,
    (__C.ADVERSARIAL_METHOD + "%.3f" % __C.ADVERSARIAL_NOISE_RATE)
    if __C.ADVERSARIAL_NOISE_RATE > 1e-5
    else "noadv",
    "lnclean" if __C.NOISE_RATE < 1e-6 else "ln%.1f" % __C.NOISE_RATE,
    "fclean" if __C.GAUSSIAN_NOISE_RATE < 1e-6 else "fn%.1e" % __C.GAUSSIAN_NOISE_RATE,
    str(__C.TIMESTAMP),
)

__C.WRITER = SummaryWriter(comment=__C.MODEL_NAME)

# --------------- optimizer configurations --------------- #

__C.OPTIMIZER, __C.SCHEDULER = get_optimizer(
    __C.MODEL, __C.OPTIMIZER_TYPE, __C.LR, __C.BEG_EPOCHS, __C.T_0, __C.T_MULT,
)

# ---------------- milestone optional load --------------- #

__C.MODEL_PATH = os.path.join("model", __C.MODEL_NAME)
check_dir(__C.MODEL_PATH)

if __C.MILESTONE_PATH is not None:
    load_model(__C.MODEL, __C.OPTIMIZER, __C.MILESTONE_PATH, __C.BEG_EPOCHS)
else:
    init_weights(__C.MODEL)

# --------------------- print configs -------------------- #

ic(__C)
# config_str = tabulate(__C.items())
# print(config_str)
# with open(__C.TABULATE_PATH, "w") as f:
#     f.write(config_str)
if __C.DEBUG:
    exit(0)

# ----------------- train/eval functions ----------------- #


def train(config, current_epoch: int, loader, train: bool = True):
    r"""
    Train 1 epoch
    """
    if train:
        __C.MODEL.train()
    else:
        __C.MODEL.eval()

    __C.OPTIMIZER.zero_grad()

    # forward pass metrics
    (
        total_loss,
        total_acc,
        total_oacc,
        total_real_confidence,
        total_noise_confidence,
        total_real_entropy,
        total_noise_entropy,
        total_lid,
    ) = [Counter() for _ in range(8)]
    # adversarial pass metrics
    (
        adv_total_loss,
        adv_total_acc,
        adv_total_oacc,
        adv_total_real_confidence,
        adv_total_noise_confidence,
        adv_total_real_entropy,
        adv_total_noise_entropy,
    ) = [Counter() for _ in range(7)]
    with torch.set_grad_enabled(train):
        # for i, batch in tqdm(enumerate(loader), total=len(loader)):
        for i, batch in enumerate(loader):
            # torch.cuda.empty_cache()
            batch = process_batch(batch, __C.DEVICE, __C.PARALLEL, __C.DATASET_TYPE)
            __C.OPTIMIZER.zero_grad()
            data_size = batch.x[batch.mask].shape[0]
            # forward pass
            # with torch.autograd.set_detect_anomaly(True):pip
            (
                loss,
                x,
                real_conf,
                noise_conf,
                correct,
                original_correct,
                real_ent,
                noise_ent,
                # lid
            ) = __C.MODEL(batch, config)
            lid = __C.MODEL.lid
            loss = loss.mean()
            real_conf = real_conf.mean()
            noise_conf = noise_conf.mean()
            real_ent = real_ent.mean()
            noise_ent = noise_ent.mean()
            acc = correct.sum() / data_size
            oacc = original_correct.sum() / data_size
            if train:
                # backward pass
                loss.backward()
                nn.utils.clip_grad_value_(__C.MODEL.parameters(), 1e2)
                __C.OPTIMIZER.step()
                __C.SCHEDULER.step()
                current_lr = __C.OPTIMIZER.param_groups[0]["lr"]
                __C.OPTIMIZER.zero_grad()
            if current_epoch % __C.REPORT_ITERATIONS == 0:
                if train:
                    ic(current_lr)
                print(
                    colorama.Fore.MAGENTA
                    + "[%d%s/%d]LOSS: %.2e, NoisyACC: %.2f%%, CleanACC: %.2f%%, LID: %.2f, NoisyCONF: %.2f, CleanCONF: %.2f, NoisyENT: %.2f, CleanENT:%.2f"
                    % (
                        epoch,
                        "t" if train else "e",
                        i,
                        loss.detach().item(),
                        100.0 * acc.detach().item(),  # comparint to noisy labels
                        100.0 * oacc.detach().item(),  # comparing to real labels
                        lid.detach().item(),
                        noise_conf.detach().item(),
                        real_conf.detach().item(),
                        noise_ent.detach().item(),
                        real_ent.detach().item(),
                    )
                )

            # adversarial pass, grad required
            with torch.set_grad_enabled(True):
                if __C.ADVERSARIAL_NOISE_RATE > 1e-5:
                    if __C.ADVERSARIAL_METHOD == "FGSM":
                        adv_batch = copy_batch(batch)
                        adv_batch.x.requires_grad = True
                        loss, *_ = __C.MODEL(adv_batch, config)
                        loss = loss.mean()
                        loss.backward()
                        adv = adv_batch.x
                        norm = adv.abs().max(dim=-1)[0].mean()  # mean max
                        # ic(norm, adv.abs().max())
                        vadv = (
                            torch.sgn(adv.grad) * norm * __C.ADVERSARIAL_NOISE_RATE
                        )  # adv batch
                        adv = vadv + adv
                        # ic(adv_batch.y, adv_batch.y.shape)
                        adv_batch.x = adv
                        with torch.no_grad():
                            (
                                adv_loss,
                                _,
                                adv_real_conf,
                                adv_noise_conf,
                                adv_correct,
                                adv_original_correct,
                                adv_real_ent,
                                adv_noise_ent,
                            ) = __C.MODEL(adv_batch, config)
                        adv_loss = adv_loss.mean()
                        adv_real_conf = adv_real_conf.mean()
                        adv_noise_conf = adv_noise_conf.mean()
                        adv_real_ent = adv_real_ent.mean()
                        adv_noise_ent = adv_noise_ent.mean()
                        adv_acc = adv_correct.sum() / data_size
                        adv_oacc = adv_original_correct.sum() / data_size
                    else:
                        raise NotImplementedError
                    __C.OPTIMIZER.zero_grad()  # clean grads after adversarial pass
                    if current_epoch % __C.REPORT_ITERATIONS == 0:
                        print(
                            colorama.Fore.MAGENTA
                            + "[%d%s/%d]LOSS: %.2e, NoisyACC: %.2f%%, CleanACC: %.2f%%, NoisyCONF: %.2f, CleanCONF: %.2f, NoisyENT: %.2f, CleanENT:%.2f"
                            % (
                                epoch,
                                "t-adv" if train else "e-adv",
                                i,
                                adv_loss.detach().item(),
                                100.0 * adv_acc.detach().item(),
                                100.0 * adv_oacc.detach().item(),
                                adv_noise_conf.detach().item(),
                                adv_real_conf.detach().item(),
                                adv_noise_ent.detach().item(),
                                adv_real_ent.detach().item(),
                            )
                        )
                else:
                    # fill stats with 0.
                    (
                        adv_loss,
                        adv_real_conf,
                        adv_noise_conf,
                        adv_acc,
                        adv_oacc,
                        adv_real_ent,
                        adv_noise_ent,
                    ) = torch.zeros([7])

                # record these adversarial metrics
                with torch.no_grad():
                    adv_total_loss.add(adv_loss)
                    adv_total_acc.add(adv_acc)
                    adv_total_oacc.add(adv_oacc)
                    adv_total_real_confidence.add(adv_real_conf)
                    adv_total_noise_confidence.add(adv_noise_conf)
                    adv_total_real_entropy.add(adv_real_ent)
                    adv_total_noise_entropy.add(adv_noise_ent)

            with torch.no_grad():
                total_loss.add(loss)
                total_acc.add(acc)
                total_oacc.add(oacc)
                total_real_confidence.add(real_conf)
                total_noise_confidence.add(noise_conf)
                total_real_entropy.add(real_ent)
                total_noise_entropy.add(noise_ent)
                total_lid.add(lid)
    return (
        total_loss.mean,
        total_acc.mean,
        total_oacc.mean,
        total_real_confidence.mean,
        total_noise_confidence.mean,
        total_real_entropy.mean,
        total_noise_entropy.mean,
        total_lid.mean,
        adv_total_loss.mean,
        adv_total_acc.mean,
        adv_total_oacc.mean,
        adv_total_real_confidence.mean,
        adv_total_noise_confidence.mean,
        adv_total_real_entropy.mean,
        adv_total_noise_entropy.mean,
        None if not train else current_lr,
    )


# ------------------------- main ------------------------- #

if __name__ == "__main__":
    for epoch in range(__C.BEG_EPOCHS, __C.TOTAL_EPOCHS + 1):
        (
            train_loss,
            train_acc,
            train_oacc,
            train_real_confidence,
            train_noise_confidence,
            train_real_entropy,
            train_noise_entropy,
            train_lid,
            adv_train_loss,
            adv_train_acc,
            adv_train_oacc,
            adv_train_real_confidence,
            adv_train_noise_confidence,
            adv_train_real_entropy,
            adv_train_noise_entropy,
            current_lr,
        ) = train(__C, epoch, loader=__C.TRAIN_LOADER, train=True)
        (
            test_acc,
            test_loss,
            test_oacc,
            test_real_confidence,
            test_noise_confidence,
            test_real_entropy,
            test_noise_entropy,
            test_lid,
            adv_test_loss,
            adv_test_acc,
            adv_test_oacc,
            adv_test_real_confidence,
            adv_test_noise_confidence,
            adv_test_real_entropy,
            adv_test_noise_entropy,
            _,
        ) = train(__C, epoch, loader=__C.TEST_loader, train=False)
        # ---------------------- save model ---------------------- #
        if epoch % __C.MILESTONE_SAVE_EPOCHS == 0 and epoch != 0:
            if __C.PARALLEL:
                model_to_save = __C.MODEL.module
            else:
                model_to_save = __C.MODEL
            torch.save(
                model_to_save.state_dict(),
                os.path.join(__C.MODEL_PATH, "model-%d.save" % (epoch)),
            )
            torch.save(
                __C.OPTIMIZER.state_dict(),
                os.path.join(__C.MODEL_PATH, "opt-%d.save" % (epoch)),
            )
            torch.save(
                model_to_save.state_dict(),
                os.path.join(__C.MODEL_PATH, "model-latest.save"),
            )
            torch.save(
                __C.OPTIMIZER.state_dict(),
                os.path.join(__C.MODEL_PATH, "opt-latest.save"),
            )

        # ------------------ log to tensorboard ------------------ #
        record_dict = {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_oacc": train_oacc,
            "train_real_confidence": train_real_confidence,
            "train_noise_confidence": train_noise_confidence,
            "train_real_entropy": train_real_entropy,
            "train_noise_entropy": train_noise_entropy,
            "train_lid": train_lid,
            "adv_train_loss": adv_train_loss,
            "adv_train_acc": adv_train_acc,
            "adv_train_oacc": adv_train_oacc,
            "adv_train_real_confidence": adv_train_real_confidence,
            "adv_train_noise_confidence": adv_train_noise_confidence,
            "adv_train_real_entropy": adv_train_real_entropy,
            "adv_train_noise_entropy": adv_train_noise_entropy,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_oacc": test_oacc,
            "test_lid": test_lid,
            "test_real_confidence": test_real_confidence,
            "test_noise_confidence": test_noise_confidence,
            "test_real_entropy": test_real_entropy,
            "test_noise_entropy": test_noise_entropy,
            "adv_test_loss": adv_test_loss,
            "adv_test_acc": adv_test_acc,
            "adv_test_oacc": adv_test_oacc,
            "adv_test_real_confidence": adv_test_real_confidence,
            "adv_test_noise_confidence": adv_test_noise_confidence,
            "adv_test_real_entropy": adv_test_real_entropy,
            "adv_test_noise_entropy": adv_test_noise_entropy,
            "current_lr": current_lr,
        }

        for key in record_dict:
            if not isinstance(record_dict[key], dict):
                __C.WRITER.add_scalar(key, record_dict[key], epoch)
            else:
                __C.WRITER.add_scalars(key, record_dict[key], epoch)
                # add multiple records
