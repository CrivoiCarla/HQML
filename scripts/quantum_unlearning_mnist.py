import os
import math
import time
import json
import copy
import random
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Torch framework for neural networks, gradient processing, and optimization
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Scikit-learn for dataset partitions, dimension reduction, and metric calculations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA

# Torchvision for handling MNIST image extraction and data transformations
from torchvision import datasets, transforms

# PennyLane tools for building hybrid quantum-classical pipeline layers
import pennylane as qml
from pennylane.qnn import TorchLayer

# -----------------------------------------------------------------------------
# Global Config & Reproduction Setup
# -----------------------------------------------------------------------------
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Output directory for saving model weights, history files, metrics CSV, and plots
OUT_DIR = "ql_unlearning_outputs_mnist"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE_NAME = "default.qubit"     # Use statevector simulation (exact expectations) or 'default.mixed' for noisy channels
ENABLE_NOISE = False              # Set to True to enable depolarizing error simulation
NOISE_P = 0.02                    # Depolarizing probability parameter within open quantum circuits

N_QUBITS = 8                      # Number of active qubits (sets the size of PennyLane expectation output)
N_CLASSES = 10                    # MNIST has 10 classes (digits 0 through 9)
BATCH = 64                        # Standard optimization batch size
LR = 1e-3                         # Adam learning rate for model weights
EPOCHS_TEACHER = 60               # Maximum training boundary for baseline models

# >>> Select the input feature representation strategy:
USE_CNN_FRONTEND = True  # True: Image space inputs to local CNN, False: Flattened features to PCA

# -----------------------------------------------------------------------------
# Data Loader Utilities (PCA Pipeline Strategy)
# -----------------------------------------------------------------------------
def load_data(n_components=N_QUBITS, samples_per_class=100):
    """
    Loads raw MNIST digits, selects balanced 'samples_per_class' per class,
    projects flattened 28x28 pixel tensors down to 'n_components' via PCA,
    and scales components to [-pi, pi] to support quantum gates.
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data_mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data_mnist", train=False, download=True, transform=tfm)

    # Flatten coordinates and divide by 255.0 to yield normalized floats in [0, 1]
    X_train = train_ds.data.numpy().reshape(-1, 28*28) / 255.0
    y_train = train_ds.targets.numpy()
    X_test  = test_ds.data.numpy().reshape(-1, 28*28) / 255.0
    y_test  = test_ds.targets.numpy()

    X = np.vstack([X_train, X_test])
    y = np.concatenate([y_train, y_test])

    # Enforce label-balanced sampling across classes to prevent bias
    X_sel, y_sel = [], []
    rng = np.random.default_rng(SEED)
    for c in range(10):
        idx_c = np.where(y == c)[0]
        chosen = rng.choice(idx_c, size=min(samples_per_class, len(idx_c)), replace=False)
        X_sel.append(X[chosen])
        y_sel.append(y[chosen])
    X_sel = np.vstack(X_sel)
    y_sel = np.concatenate(y_sel)

    # Apply PCA dimensional reduction into target quantum register dimensions
    pca = PCA(n_components=n_components, random_state=SEED)
    X_red = pca.fit_transform(X_sel)
    # Scale coordinates to boundary interval matching Euler expectations
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_enc = scaler.fit_transform(X_red)

    print(f"[INFO] MNIST subset (PCA): {len(X_enc)} mostre totale ({samples_per_class}/clasă)")
    return X_enc, y_sel

# -----------------------------------------------------------------------------
# Data Loader Utilities (CNN Image-Space Pipeline Strategy)
# -----------------------------------------------------------------------------
def load_data_cnn(samples_per_class=100):
    """
    Loads raw 1x28x28 MNIST images and compiles balanced groups.
    Outputs normalized image arrays of shape [N, 1, 28, 28] and targets.
    """
    tfm = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root="data_mnist", train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="data_mnist", train=False, download=True, transform=tfm)

    X = torch.cat([train_ds.data, test_ds.data], dim=0).float() / 255.0  # Normalized to [0.0, 1.0]
    y = torch.cat([train_ds.targets, test_ds.targets], dim=0).long()

    # Balanced subset sampling from image sets
    X_sel, y_sel = [], []
    g = torch.Generator().manual_seed(SEED)
    for c in range(N_CLASSES):
        idx = (y == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(idx), generator=g)
        chosen = idx[perm[:samples_per_class]]
        X_sel.append(X[chosen])
        y_sel.append(y[chosen])
    X_sel = torch.cat(X_sel, dim=0).unsqueeze(1)  # Reshape to [N, 1, 28, 28] channel-first format
    y_sel = torch.cat(y_sel, dim=0)

    print(f"[INFO] MNIST subset (images): {len(X_sel)} mostre totale ({samples_per_class}/class)")
    return X_sel.numpy(), y_sel.numpy()

# -----------------------------------------------------------------------------
# PyTorch Format Adapters & Train Scenarios Partitioning
# -----------------------------------------------------------------------------
def to_tensors(X, y=None):
    """
    Encapsulates raw arrays inside standard float/long Torch tensors.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    if y is None:
        return X_t
    y_t = torch.tensor(y, dtype=torch.long)
    return X_t, y_t

def make_loaders(X_train, y_train, X_test, y_test, batch=BATCH):
    """
    Prepares train and test dataloaders used during mini-batch loop execution.
    """
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                            torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)
    return train_loader, test_loader

def split_scenarios(X, y, subset_rate=0.02):
    """
    Segregates dataset into baseline pools and unlearning scenarios:
      - Stratified split: 80% baseline train, 20% test
      - Scenario A (Forget Subset): target 2% of the training set for unlearning
      - Scenario B (Forget Class): target 100% of Class 0 in the training set for deletion
    """
    X_tr_all, X_te, y_tr_all, y_te = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    # Scenario A: Slice a small subset (2%) to act as private forget subset
    X_dr_A, X_df_A, y_dr_A, y_df_A = train_test_split(
        X_tr_all, y_tr_all, test_size=subset_rate, random_state=SEED, stratify=y_tr_all
    )
    # Scenario B: Slice Class 0 entirely as forget subset
    mask_class0 = (y_tr_all == 0)
    X_df_B = X_tr_all[mask_class0]
    y_df_B = y_tr_all[mask_class0]
    X_dr_B = X_tr_all[~mask_class0]
    y_dr_B = y_tr_all[~mask_class0]

    return (
        (X_tr_all, y_tr_all, X_te, y_te),
        (X_dr_A, y_dr_A, X_df_A, y_df_A),
        (X_dr_B, y_dr_B, X_df_B, y_df_B),
    )

# -----------------------------------------------------------------------------
# Quantum State and QNode Construction
# -----------------------------------------------------------------------------
def make_device():
    """
    Allocates quantum register devices. 
    Selects 'default.mixed' for open system depolarizing error runs when active.
    """
    if ENABLE_NOISE:
        dev = qml.device("default.mixed", wires=N_QUBITS, shots=None)
    else:
        dev = qml.device(DEVICE_NAME, wires=N_QUBITS, shots=None)
    return dev

def noise_block():
    """
    Implements depolarizing error channels over wires if noise is enabled.
    """
    if ENABLE_NOISE:
        for w in range(N_QUBITS):
            qml.DepolarizingChannel(NOISE_P, wires=w)

def qnode_def(dev):
    """
    Defines integration of backpropagation using PyTorch interface on PennyLane:
      - Feature Encoding: RX, RY, RZ Euler rotations
      - Variational block: arbitary RX, RY, RZ rotations followed by ring CZ gates
    """
    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        # inputs: 1D tensor with N_QUBITS features scaled to angle intervals
        assert inputs.shape[0] == N_QUBITS, f"Expected {N_QUBITS} features, got {inputs.shape[0]}"
        # Angle encoding mechanics
        for w, x in enumerate(inputs):
            qml.RX(x, wires=w)
            qml.RY(x, wires=w)
            qml.RZ(x, wires=w)
        noise_block()
        # L variational layers featuring controllable variables and ring topologies
        L = weights.shape[0]
        for l in range(L):
            for w in range(N_QUBITS):
                qml.RX(weights[l, 0, w], wires=w)
                qml.RY(weights[l, 1, w], wires=w)
                qml.RZ(weights[l, 2, w], wires=w)
            for w in range(N_QUBITS - 1):
                qml.CZ(wires=[w, w + 1])
            qml.CZ(wires=[N_QUBITS - 1, 0])
            noise_block()
        # Expectații Z -> N_QUBITS dimensiuni reale
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]
    return qnode

# -----------------------------------------------------------------------------
# Unlearning-Compatible Model Implementations (PCA vs CNN options)
# -----------------------------------------------------------------------------
class HybridQCNN(nn.Module):
    """
    Standard PCA-hybrid interface.
    Uses pre-reduced classical PCA input vectors:
      PCA Vector [B, N_QUBITS] -> QLayer expectvals [B, N_QUBITS] -> Linear(16) -> Relu -> Linear(10)
    """
    def __init__(self, layers=2):
        super().__init__()
        dev = make_device()
        qnode = qnode_def(dev)
        # Weights shape: (num_layers, 3 gates [RX, RY, RZ], num_qubits)
        weight_shapes = {"weights": (layers, 3, N_QUBITS)}
        self.q_layer = TorchLayer(qnode, weight_shapes)
        self.fc1 = nn.Linear(N_QUBITS, 16)
        self.fc2 = nn.Linear(16, N_CLASSES)

    def forward(self, x):
        # Obtain expectations from the simulator block
        outs = [self.q_layer(sample) for sample in x]
        q_out = torch.stack(outs)   # Shape: [B, N_QUBITS]
        q_out = torch.relu(self.fc1(q_out))
        logits = self.fc2(q_out)
        return logits

class HybridQConv(nn.Module):
    """
    CNN-hybrid image-space interface.
    Accepts raw [B, 1, 28, 28] images, maps features down to standard angle metrics using
    a CNN frontend, encodes those angles onto the qubits, and executes linear classifiers:
      Image [B, 1, 28, 28] -> CNN features [B, 32, 7, 7] -> Flat -> Linear projection -> angles [B, N_QUBITS] in [-pi, pi]
                           -> QLayer expectvals [B, N_QUBITS] -> Linear(32) -> Relu -> Linear(10)
    """
    def __init__(self, layers=2):
        super().__init__()
        # 1) Frontend CNN encoder (Accepts size 1x28x28)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 28x28 -> 16x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # -> 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # -> 32x14x14
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                                       # -> 32x7x7
        )
        self.to_angles = nn.Sequential(
            nn.Flatten(),                           # Flatten size 32 * 7 * 7 = 1568
            nn.Linear(32*7*7, N_QUBITS)             # Maps to active qubit angles
        )

        # 2) Variational PennyLane Quantum Layer
        dev = make_device()
        qnode = qnode_def(dev)
        weight_shapes = {"weights": (layers, 3, N_QUBITS)}
        self.q_layer = TorchLayer(qnode, weight_shapes)

        # 3) Post-quantum classical head
        self.fc1 = nn.Linear(N_QUBITS, 32)
        self.fc2 = nn.Linear(32, N_CLASSES)

    def forward(self, x):
        """
        Input x: [B, 1, 28, 28] Normalized float images.
        """
        # CNN Feature extraction
        feat = self.cnn(x)                          # Shape: [B, 32, 7, 7]
        # Map features to raw angle parameters
        angles = self.to_angles(feat)               # Shape: [B, N_QUBITS]
        # Squish outputs to interval [-pi, pi] for angle encoding
        angles = torch.tanh(angles) * math.pi

        # Process through the PennyLane simulation layer
        outs = [self.q_layer(sample) for sample in angles]  
        q_out = torch.stack(outs)                            # Shape: [B, N_QUBITS]

        x = torch.relu(self.fc1(q_out))
        logits = self.fc2(x)
        return logits

# -----------------------------------------------------------------------------
# Model Persistence & Performance Evaluation Helpers
# -----------------------------------------------------------------------------
def save_model(model, path, extra: Dict = None):
    """
    Saves state dictionary supplemented with custom training metadata.
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "extra": extra or {}
    }
    torch.save(payload, path)

def load_model(path, layers=2):
    """
    Restores either CNN-hybrid or PCA-hybrid parameters depending on USE_CNN_FRONTEND.
    """
    model = HybridQConv(layers=layers) if USE_CNN_FRONTEND else HybridQCNN(layers=layers)
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state_dict"])
    return model, payload.get("extra", {})

def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Applies standard Softmax transformation over output logits to produce probability tables.
    """
    return torch.softmax(logits, dim=1)

def evaluate_acc(model, loader):
    """
    Loops over evaluation streams, computing classifier accuracy.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    return correct / total if total else 0.0

def get_probs(model, loader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes/returns concatenated probability distributions and true labels.
    """
    model.eval()
    probs_list, y_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = softmax_probs(model(xb)).cpu().numpy()
            probs_list.append(probs)
            y_list.append(yb.cpu().numpy())
    return np.vstack(probs_list), np.concatenate(y_list)

def kl_divergence(P: np.ndarray, Q: np.ndarray, eps=1e-8) -> float:
    """
    Computes Kullback-Leibler (KL) divergence between unlearned distribution P and oracle target Q.
    """
    P = np.clip(P, eps, 1)
    Q = np.clip(Q, eps, 1)
    return float(np.mean(np.sum(P * (np.log(P) - np.log(Q)), axis=1)))

def js_divergence(P: np.ndarray, Q: np.ndarray, eps=1e-8) -> float:
    """
    Computes symmetric Jensen-Shannon divergence over P and Q using a cached mean matrix M.
    """
    M = 0.5 * (P + Q)
    def _kl(A, B):
        A = np.clip(A, eps, 1); B = np.clip(B, eps, 1)
        return np.sum(A * (np.log(A) - np.log(B)), axis=1)
    return float(0.5 * np.mean(_kl(P, M)) + 0.5 * np.mean(_kl(Q, M)))

def agreement_rate(model_a, model_b, loader) -> float:
    """
    Measures ratio of test elements yielding the exact same output class decisions between models.
    """
    model_a.eval(); model_b.eval()
    pa, pb = [], []
    with torch.no_grad():
        for xb, _ in loader:
            pa.append(model_a(xb).argmax(1).cpu().numpy())
            pb.append(model_b(xb).argmax(1).cpu().numpy())
    pa = np.concatenate(pa); pb = np.concatenate(pb)
    return float(np.mean(pa == pb))

def mia_auc_confidence(model, members_loader, nonmembers_loader) -> float:
    """
    Confidence-based Membership Inference Attack (MIA) evaluation score.
    Higher values represent susceptibility to training sample trace leaks.
    """
    def scores(loader):
        s = []
        with torch.no_grad():
            for xb, _ in loader:
                probs = softmax_probs(model(xb)).cpu().numpy()
                s.extend(np.max(probs, axis=1))
        return np.array(s)

    s_members = scores(members_loader)
    s_nonmembers = scores(nonmembers_loader)
    labels = np.concatenate([np.ones_like(s_members), np.zeros_like(s_nonmembers)])
    scores_all = np.concatenate([s_members, s_nonmembers])
    try:
        auc = roc_auc_score(labels, scores_all)
    except Exception:
        auc = float("nan")
    return float(auc)

# =============================================================================
# Early Stopping Tracking Mechanics
# =============================================================================
class EarlyStopper:
    """
    Halts optimization steps when target metrics fail to improve for consecutive epochs.
    """
    def __init__(self, patience=6, min_delta=0.0, best_is_max=True):
        self.patience = patience
        self.min_delta = min_delta
        self.best_is_max = best_is_max
        self.best_val = None
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.best_state = None

    def _is_better(self, val):
        if self.best_val is None:
            return True
        if self.best_is_max:
            return (val - self.best_val) > self.min_delta
        else:
            return (self.best_val - val) > self.min_delta

    def step(self, val, model, epoch):
        if self._is_better(val):
            self.best_val = val
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            improved = True
        else:
            self.epochs_no_improve += 1
            improved = False
        stop = self.epochs_no_improve >= self.patience
        return stop, improved

def _save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def train_with_es(
    model,
    epochs,
    eval_loader,
    tag,
    train_epoch_fn,  # closure: train one epoch
    best_is_max=True,
    patience=6,
    min_delta=0.0,
    save_every_epoch=True,
):
    t0 = time.perf_counter()
    stopper = EarlyStopper(patience=patience, min_delta=min_delta, best_is_max=best_is_max)

    best_path = os.path.join(OUT_DIR, f"{tag}_best.pth")
    per_epoch_paths = []

    for ep in range(1, epochs + 1):
        train_epoch_fn(ep)

        val_acc = evaluate_acc(model, eval_loader)

        if save_every_epoch:
            ep_path = os.path.join(OUT_DIR, f"{tag}_ep{ep}.pth")
            save_model(model, ep_path, extra={"epoch": ep, "val_acc": val_acc, "tag": tag})
            per_epoch_paths.append(ep_path)

        stop, improved = stopper.step(val_acc, model, ep)
        print(f"[{tag}] Ep {ep:02d} | ValAcc {val_acc*100:.2f}% "
              f"{'(best)' if improved else ''} | no_improve={stopper.epochs_no_improve}/{patience}")
        if stop:
            print(f"[{tag}] Early stopping {ep} (best @ {stopper.best_epoch}).")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
        save_model(model, best_path, extra={"epoch": stopper.best_epoch, "best_val_acc": stopper.best_val, "tag": tag})

    elapsed = time.perf_counter() - t0
    log = {
        "tag": tag,
        "epochs_requested": epochs,
        "epochs_ran": ep,
        "stopped_early": (ep < epochs),
        "best_epoch": stopper.best_epoch,
        "best_val_acc": stopper.best_val,
        "patience": patience,
        "min_delta": min_delta,
        "elapsed_sec": elapsed,
        "per_epoch_checkpoints": per_epoch_paths,
        "best_checkpoint": best_path,
    }
    _save_json(os.path.join(OUT_DIR, f"{tag}_trainlog.json"), log)
    print(f"[{tag}] Train finished in {elapsed:.2f}s. Best val acc {100*(stopper.best_val or 0):.2f}% @ epoch {stopper.best_epoch}. "
          f"Logs: {os.path.join(OUT_DIR, f'{tag}_trainlog.json')}")
    return model, log

# ----------------------------
# Teacher training (with ES)
# ----------------------------
def train_teacher(X_train, y_train, X_test, y_test, layers=2, epochs=EPOCHS_TEACHER, lr=LR, tag="teacher",
                  patience=6, min_delta=0.0):
    model = HybridQConv(layers=layers) if USE_CNN_FRONTEND else HybridQCNN(layers=layers)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    train_loader, eval_loader = make_loaders(X_train, y_train, X_test, y_test, batch=BATCH)

    def _one_epoch(ep):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[{tag}] Ep {ep:02d} TrainLoss {total_loss:.4f}")

    model, _ = train_with_es(
        model=model,
        epochs=epochs,
        eval_loader=eval_loader,
        tag=tag,
        train_epoch_fn=_one_epoch,
        best_is_max=True,
        patience=patience,
        min_delta=min_delta,
        save_every_epoch=True,
    )

    best_path = os.path.join(OUT_DIR, f"{tag}_best.pth")
    model_best, _ = load_model(best_path, layers=layers)
    return model_best, (train_loader, eval_loader), best_path

# -----------------------------------------------------------------------------
# CF-k Selective Layer Group Freeze Helpers
# -----------------------------------------------------------------------------
def get_layer_groups_for_freeze(model):
    """
    Returns an ordered list of layer groups/modules for CF-k selective freezing,
    sorted starting from input layer to output classification projections.
    This abstraction allows CF-k to operate seamlessly whether we use the CNN frontend or the PCA frontend.
    """
    if isinstance(model, HybridQConv):
        # Ordered list for CNN model: cnn encoder, projection linear, quantum encoder, fc1, fc2
        return [model.cnn, model.to_angles, model.q_layer, model.fc1, model.fc2]
    else:
        # Ordered list for PCA model: quantum encoder, fc1, fc2
        return [model.q_layer, model.fc1, model.fc2]

# -----------------------------------------------------------------------------
# Machine Unlearning Core Algorithms
# -----------------------------------------------------------------------------

# --- 1) Gradient Ascent (GA) ---
def method_gradient_ascent(model, forget_loader, epochs=10, lr=LR, tag="GA",
                           eval_loader=None, patience=6, min_delta=0.0):
    """
    Executes raw class/subset unlearning via Gradient Ascent.
    We negate Cross Entropy Loss over forget batch entries (Df) to force prediction divergence:
      Loss = -CE(model(X_f), y_f)
    """
    model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    def _one_epoch(ep):
        model.train()
        for xb, yb in forget_loader:
            opt.zero_grad()
            loss = -criterion(model(xb), yb)  # Maximizing prediction error on forget elements
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 2) Fisher Unlearning (EWC-Inspired Quadratic Penalty) ---
def method_fisher_unlearning(model, forget_loader, lambda_ewc=10.0, epochs=10, lr=LR, tag="Fisher",
                             eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements Fisher Unlearning.
    Forces predicted values over forget subset elements to diverge, while restricting parameter shift
    weighted by diagonal elements of the computed Fisher information matrix over the forget set:
      Loss = -CE(model(X_f), y_f) + (\lambda / 2) * \sum_j F_j * (\theta_j - \theta_{original, j})^2
    """
    model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
    fisher = compute_fisher(model, forget_loader, criterion)

    def _one_epoch(ep):
        model.train()
        for xb, yb in forget_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = -criterion(logits, yb)
            # Apply quadratic penalty to prevent unlearned parameters from drifting in high-Fisher channels
            for (n, p) in model.named_parameters():
                if p.requires_grad:
                    loss = loss + (lambda_ewc / 2.0) * torch.sum(fisher[n] * (p - old_params[n]) ** 2)
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 3) NegGrad+ (Dual-Optimization retain vs forget balance) ---
def method_neggrad_plus(model, retain_loader, forget_loader, epochs=10, lr=LR, alpha=1.0, tag="NegGradPlus",
                        eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements NegGrad+ unlearning.
    Jointly minimizes performance error over the retain loader while maximizing error on forget set elements:
      Loss = CE(model(X_r), y_r) - alpha * CE(model(X_f), y_f)
    """
    model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    def _one_epoch(ep):
        model.train()
        for (xb_r, yb_r), (xb_f, yb_f) in zip(retain_loader, forget_loader):
            opt.zero_grad()
            out_r = model(xb_r)
            out_f = model(xb_f)
            loss = criterion(out_r, yb_r) - alpha * criterion(out_f, yb_f)
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 4) CF-k (Selective Layer Freezing / Classic Freezing) ---
def method_cf_k(model, forget_loader, k=1, epochs=10, lr=LR, tag="CFk",
                eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements CF-k.
    Locks the parameter gradients belonging to the first k layer groups in place, 
    and drives gradient ascent on the forget set exclusively within the subsequent layers.
    Selects layer groups using the `get_layer_groups_for_freeze` helper.
    """
    model = copy.deepcopy(model)
    groups = get_layer_groups_for_freeze(model)
    # Freeze layers up to index k
    for i, grp in enumerate(groups):
        for p in grp.parameters() if hasattr(grp, "parameters") else []:
            p.requires_grad = not (i < k)

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def _one_epoch(ep):
        model.train()
        for xb, yb in forget_loader:
            opt.zero_grad()
            logits = model(xb)
            loss = -criterion(logits, yb)
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 5) EU-k (Encoder Re-initialization & Fine-Tuning) ---
def method_eu_k(model, retain_loader, k=1, epochs=10, lr=LR, tag="EUk",
                eval_loader=None, patience=6, min_delta=0.0):
    """
    Reinitializes the last k classical layers from scratch, then fine-tunes parameters over Dr.
    This selectively deletes learned boundaries of the upper classification head.
      - k = 1: Reinitialize `fc2`
      - k = 2: Reinitialize both `fc1` and `fc2`
    """
    model = copy.deepcopy(model)
    # Reinitialize tail classical modules
    if k >= 1:
        if hasattr(model, "fc2"):
            model.fc2 = nn.Linear(model.fc1.out_features, N_CLASSES)
    if k >= 2:
        if hasattr(model, "fc1"):
            model.fc1 = nn.Linear(N_QUBITS, model.fc2.in_features if hasattr(model.fc2, "in_features") else 32)
        if hasattr(model, "fc2"):
            model.fc2 = nn.Linear(model.fc1.out_features, N_CLASSES)

    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    def _one_epoch(ep):
        model.train()
        for xb, yb in retain_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 6) Certified Unlearning (Noisy Fine-tuning) ---
def method_certified_unlearning(model, retain_loader, epochs=20, lr=LR,
                                sigma=0.05, clip_norm=1.0, tag="Certified",
                                eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements Certified Unlearning.
    Slightly perturbs the model over the retain set by executing clipped gradients
    with added Gaussian noise, borrowing foundations from Differential Privacy (DP-SGD).
    This mathematically bounds the parameter impact of any individual training element.
    """
    model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    def _one_epoch(ep):
        model.train()
        for xb, yb in retain_loader:
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            # Stabilize parameters sensitivity using gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            # Add Gaussian noise vectors directly onto parameters gradients
            for p in model.parameters():
                if p.grad is not None:
                    p.grad += sigma * torch.randn_like(p.grad)
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# --- 7) Q-MUL (Quantum Machine Unlearning with Similar Labels) ---
def build_similar_label_map(teacher_model, data_X, data_y) -> Dict[int, int]:
    """
    Determines close class neighbors by finding maximum Cosine Similarity
    between mean logit centroids of classes.
    """
    teacher_model.eval()
    X_t = torch.tensor(data_X, dtype=torch.float32)
    y_t = torch.tensor(data_y, dtype=torch.long)
    with torch.no_grad():
        logits = teacher_model(X_t).cpu().numpy()

    class_means = []
    for c in range(N_CLASSES):
        mask = (y_t.numpy() == c)
        if np.sum(mask) == 0:
            class_means.append(np.zeros((N_CLASSES,), dtype=np.float32))
        else:
            class_means.append(logits[mask].mean(axis=0))
    class_means = np.stack(class_means, axis=0)

    def cos(a, b):
        na = np.linalg.norm(a) + 1e-12
        nb = np.linalg.norm(b) + 1e-12
        return float(np.dot(a, b) / (na * nb))

    mapping = {}
    for c in range(N_CLASSES):
        best = None
        for d in range(N_CLASSES):
            if d == c:
                continue
            s = cos(class_means[c], class_means[d])
            if best is None or s > best[0]:
                best = (s, d)
        mapping[c] = best[1] if best else c
    return mapping

def method_qmul(teacher_model, retain_loader, forget_loader, epochs=20,
                lr=LR, alpha=1.0, tag="Q-MUL",
                eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements Q-MUL inspired unlearning.
    Rather than executing generic gradient ascent, Q-MUL maps forget set samples (Df) 
    towards their mathematically 'most similar classes' (distractor targets).
    Furthermore, to balance unlearning updates without destroying model performance,
    it dynamically scales the loss components using gradient norms estimated on
    the retain (Lr) and forget (Lf) batches separately:
      w = (||grad_retain|| / (||grad_forget|| + eps)) * alpha
      Loss = Lr - w * Lf
    """
    model = copy.deepcopy(teacher_model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    def stack_loader(loader):
        Xs, Ys = [], []
        for xb, yb in loader:
            Xs.append(xb); Ys.append(yb)
        return torch.cat(Xs).numpy(), torch.cat(Ys).numpy()
    Xr, yr = stack_loader(retain_loader)
    sim_map = build_similar_label_map(teacher_model, Xr, yr)

    eps = 1e-12
    def _one_epoch(ep):
        model.train()
        for (xb_r, yb_r), (xb_f, yb_f) in zip(retain_loader, forget_loader):
            # similar-labels pentru forget batch
            yb_f_sim = yb_f.clone()
            for i in range(yb_f_sim.shape[0]):
                yb_f_sim[i] = sim_map[int(yb_f_sim[i].item())]

            # gradient norms
            opt.zero_grad()
            Lr = criterion(model(xb_r), yb_r)
            Lr.backward(retain_graph=True)
            gnorm_r = torch.sqrt(sum([(p.grad.detach()**2).sum()
                                      for p in model.parameters() if p.grad is not None]) + eps).item()

            model.zero_grad(set_to_none=True)
            Lf = criterion(model(xb_f), yb_f_sim)
            Lf.backward(retain_graph=True)
            gnorm_f = torch.sqrt(sum([(p.grad.detach()**2).sum()
                                      for p in model.parameters() if p.grad is not None]) + eps).item()

            w = (gnorm_r / (gnorm_f + eps)) * alpha
            model.zero_grad(set_to_none=True)
            loss = Lr - w * Lf
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# ----------------------------
# SCRUB Unlearning Method (Teacher-Student Distillation Setup)
# ----------------------------
def kl_torch(p_teacher: torch.Tensor, p_student: torch.Tensor, eps=1e-8):
    """
    Computes PyTorch tensor-based KL Divergence.
    """
    p_teacher = torch.clamp(p_teacher, eps, 1.0)
    p_student = torch.clamp(p_student, eps, 1.0)
    return torch.sum(p_teacher * (torch.log(p_teacher) - torch.log(p_student)), dim=1).mean()

def method_scrub(model_teacher, retain_loader, forget_loader, epochs=50, lr=LR, lam_r=1.0, lam_f=1.0, tag="SCRUB",
                 rewind=False, eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements SCRUB Unlearning.
    Distillation alignment:
      - Obey: Minimize student prediction KL-divergence from teacher on retain set (lam_r * L_obey)
      - Disobey: Maximize student prediction KL-divergence from teacher on forget set (-lam_f * L_disobey)
    """
    student = copy.deepcopy(model_teacher)
    opt = optim.Adam(student.parameters(), lr=lr)

    best_score = -1e9
    best_path_rewind = os.path.join(OUT_DIR, f"{tag}_rewind_best.pth")

    def _one_epoch(ep):
        nonlocal best_score
        student.train()
        for (xb_r, yb_r), (xb_f, yb_f) in zip(retain_loader, forget_loader):
            opt.zero_grad()
            with torch.no_grad():
                pt_r = softmax_probs(model_teacher(xb_r))
                pt_f = softmax_probs(model_teacher(xb_f))
            ps_r = softmax_probs(student(xb_r))
            ps_f = softmax_probs(student(xb_f))

            L_obey = kl_torch(pt_r, ps_r)       # minimize on retain
            L_disobey = kl_torch(pt_f, ps_f)    # maximize on forget

            loss = lam_r * L_obey - lam_f * L_disobey
            loss.backward()
            opt.step()

        if rewind:
            score = (-L_obey.item()) + (L_disobey.item())
            if score > best_score:
                best_score = score
                save_model(student, best_path_rewind, extra={"epoch": ep, "score": score, "tag": tag})

    student, _ = train_with_es(student, epochs, eval_loader, tag, _one_epoch,
                               best_is_max=True, patience=patience, min_delta=min_delta)
    return student

# ----------------------------
# LCA Unlearning Method (Label Complement Augmentation Setup)
# ----------------------------
from itertools import cycle

def make_label_complement_loader(forget_loader, n_classes=N_CLASSES, batch=BATCH, shuffle=True):
    """
    Transforms each forget sample (x, y) into multiple distinct labels (x, c) 
    for every available incorrect class index c != y.
    """
    Xs, Ys = [], []
    for xb, yb in forget_loader:
        Xs.append(xb)
        Ys.append(yb)
    if len(Xs) == 0:
        raise ValueError(" ! forget_loader")
    X = torch.cat(Xs)                          # [N, D...]
    y = torch.cat(Ys)                          # [N]

    N = X.shape[0]
    all_classes = torch.arange(n_classes).unsqueeze(0).repeat(N, 1)   # [N, C]
    mask = (all_classes != y.unsqueeze(1))                            # [N, C]
    y_comp = all_classes[mask].reshape(-1)                            # [N*(C-1)]

    # Replicity expansion mapping matching N*(C-1) shape dimensions
    reps = n_classes - 1
    X_rep = X.unsqueeze(1).repeat(1, reps, *([1] * (X.dim()-1))).reshape(-1, *X.shape[1:])

    ds = TensorDataset(X_rep.float(), y_comp.long())
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)

def method_label_complement_augmentation(teacher_model,
                                         retain_loader,
                                         forget_loader,
                                         epochs=10,
                                         lr=LR,
                                         beta=1.0,
                                         tag="LCA",
                                         eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements Label Complement Augmentation (LCA).
    Fine-tunes the model jointly with the standard Cross Entropy Loss on the retain set (D_r)
    and complemented class labels on the forget set (D_c), diffusing the confidence profile:
      Loss = CE(model(X_r), y_r) + beta * CE(model(X_c), y_c)
    """
    model = copy.deepcopy(teacher_model)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    comp_loader = make_label_complement_loader(forget_loader, n_classes=N_CLASSES, batch=BATCH, shuffle=True)

    def _one_epoch(ep):
        model.train()
        comp_iter = iter(comp_loader)
        for xb_r, yb_r in retain_loader:
            try:
                xb_c, yb_c = next(comp_iter)
            except StopIteration:
                comp_iter = iter(comp_loader)
                xb_c, yb_c = next(comp_iter)

            opt.zero_grad()
            Lr = criterion(model(xb_r), yb_r)
            Lc = criterion(model(xb_c), yb_c)
            loss = Lr + beta * Lc
            loss.backward()
            opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# ----------------------------
# ADV-UNIFORM Unlearning Method (Adversarial Input Perturbations with FGSM)
# ----------------------------
def fgsm_on_inputs(model, xb, eps=0.1, y_target_uniform=True):
    """
    Performs Fast Gradient Sign Method (FGSM) to adversarially perturb inputs.
    """
    xb = xb.clone().detach().requires_grad_(True)
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)
    if y_target_uniform:
        U = torch.full_like(probs, 1.0 / probs.shape[1])
        loss = torch.sum(probs * (torch.log(probs + 1e-8) - torch.log(U + 1e-8)), dim=1).mean()
    else:
        loss = -torch.distributions.Categorical(probs).entropy().mean()
    loss.backward()
    x_adv = xb + eps * xb.grad.sign()
    #  PCA: clamp  [-π,π]; images: clamp [0,1].
    if USE_CNN_FRONTEND:
        return torch.clamp(x_adv, 0.0, 1.0).detach()
    else:
        return torch.clamp(x_adv, -math.pi, math.pi).detach()

def method_adv_uniform(teacher_model, retain_loader, forget_loader,
                       epochs=10, lr=LR, lam=1.0, eps=0.1, tag="ADVUNI",
                       eval_loader=None, patience=6, min_delta=0.0):
    """
    Implements Adversarial Uniform (ADV-UNIFORM) unlearning.
    First constructs adversarial inputs x_adv from forget samples via FGSM, then minimizes
    retain set cross-entropy combined with a penalty steering predictions on x_adv towards uniform distribution:
      Loss = CE(model(X_r), y_r) + lam * KL(model(X_adv) || Uniform)
    """
    model = copy.deepcopy(teacher_model)
    ce = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    U = torch.full((N_CLASSES,), 1.0 / N_CLASSES)

    def _one_epoch(ep):
        model.train()
        for (xb_r, yb_r), (xb_f, _) in zip(retain_loader, forget_loader):
            x_adv = fgsm_on_inputs(model, xb_f, eps=eps)
            opt.zero_grad()
            Lr = ce(model(xb_r), yb_r)
            pf = torch.softmax(model(x_adv), dim=1)
            L_u = torch.sum(pf * (torch.log(pf + 1e-8) - torch.log(U.to(pf.device))), dim=1).mean()
            (Lr + lam * L_u).backward(); opt.step()

    model, _ = train_with_es(model, epochs, eval_loader, tag, _one_epoch,
                             best_is_max=True, patience=patience, min_delta=min_delta)
    return model

# -----------------------------------------------------------------------------
# Comprehensive Evaluation Pipeline Wrapper and Experiment Pipeline
# -----------------------------------------------------------------------------
def evaluate_suite(method_name, model_unlearned, model_teacher, model_oracle,
                   loaders_dict, scenario_tag, csv_rows: List[Dict]):
    """
    Fully evaluates the unlearned student model across relevant partitions and indicators:
      1. Classification accuracy on Retain, Forget, and Test pools.
      2. Performance drop/gain metrics relative to pre-trained original teacher state.
      3. Similarity checks (KL divergence, JS divergence, decision agreement rate)
         benchmarked against the retrained-from-scratch oracle baseline.
      4. Membership Inference Attack (MIA) resilience analysis.
      5. UQI (Unlearning Quality Index) equation score calculation.
    """
    retain_loader = loaders_dict["retain"]
    forget_loader = loaders_dict["forget"]
    test_loader = loaders_dict["test"]

    acc_r_u = evaluate_acc(model_unlearned, retain_loader)
    acc_f_u = evaluate_acc(model_unlearned, forget_loader)
    acc_t_u = evaluate_acc(model_unlearned, test_loader)

    acc_r_o = evaluate_acc(model_teacher, retain_loader)
    acc_f_o = evaluate_acc(model_teacher, forget_loader)
    acc_t_o = evaluate_acc(model_teacher, test_loader)

    forget_drop = acc_f_o - acc_f_u
    retain_drop = acc_r_o - acc_r_u
    test_drop   = acc_t_o - acc_t_u

    probs_u_r, _ = get_probs(model_unlearned, retain_loader)
    probs_o_r, _ = get_probs(model_oracle,   retain_loader)

    probs_u_t, _ = get_probs(model_unlearned, test_loader)
    probs_o_t, _ = get_probs(model_oracle,   test_loader)

    kl_r = kl_divergence(probs_u_r, probs_o_r)
    js_r = js_divergence(probs_u_r, probs_o_r)
    kl_t = kl_divergence(probs_u_t, probs_o_t)
    js_t = js_divergence(probs_u_t, probs_o_t)

    agree_r = agreement_rate(model_unlearned, model_oracle, retain_loader)
    agree_t = agreement_rate(model_unlearned, model_oracle, test_loader)

    mia_auc = None
    if "mia_nonmembers" in loaders_dict and loaders_dict["mia_nonmembers"] is not None:
        mia_auc = mia_auc_confidence(model_unlearned, forget_loader, loaders_dict["mia_nonmembers"])

    uqi = forget_drop - 0.5 * (retain_drop + test_drop)

    row = {
        "scenario": scenario_tag,
        "method": method_name,
        "acc_retain_unlearn": acc_r_u,
        "acc_forget_unlearn": acc_f_u,
        "acc_test_unlearn": acc_t_u,
        "acc_retain_orig": acc_r_o,
        "acc_forget_orig": acc_f_o,
        "acc_test_orig": acc_t_o,
        "forget_drop": forget_drop,
        "retain_drop": retain_drop,
        "test_drop": test_drop,
        "UQI": uqi,
        "KL_retain": kl_r,
        "JS_retain": js_r,
        "KL_test": kl_t,
        "JS_test": js_t,
        "Agree_retain": agree_r,
        "Agree_test": agree_t,
        "MIA_AUC": mia_auc
    }
    csv_rows.append(row)
    print(f"[{scenario_tag} | {method_name}] "
          f"Ret {acc_r_u:.3f} For {acc_f_u:.3f} Test {acc_t_u:.3f} | "
          f"UQI {uqi:.3f} | KLr {kl_r:.3f} JSt {js_t:.3f} | AgreeT {agree_t:.3f} | MIA {mia_auc}")

# ----------------------------
# Main experiment runner
# ----------------------------
def run_experiment():
    # 1) Date
    if USE_CNN_FRONTEND:
        X, y = load_data_cnn(samples_per_class=100)       # imagini [N,1,28,28]
    else:
        X, y = load_data(n_components=N_QUBITS, samples_per_class=100)  # vectori [N,N_QUBITS]

    (X_tr_all, y_tr_all, X_te, y_te), (X_dr_A, y_dr_A, X_df_A, y_df_A), (X_dr_B, y_dr_B, X_df_B, y_df_B) = \
        split_scenarios(X, y, subset_rate=0.02)

    # 2) Profesorul cu Early Stopping
    teacher, (train_loader_full, test_loader_global), teacher_path = train_teacher(
        X_train=X_tr_all, y_train=y_tr_all, X_test=X_te, y_test=y_te, tag="teacher_full",
        patience=8, min_delta=0.0
    )

    # 3) Loadere per scenariu
    retain_loader_A = DataLoader(TensorDataset(torch.tensor(X_dr_A, dtype=torch.float32),
                                               torch.tensor(y_dr_A, dtype=torch.long)), batch_size=BATCH, shuffle=True)
    forget_loader_A = DataLoader(TensorDataset(torch.tensor(X_df_A, dtype=torch.float32),
                                               torch.tensor(y_df_A, dtype=torch.long)), batch_size=BATCH, shuffle=True)

    test_loader = DataLoader(TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                                           torch.tensor(y_te, dtype=torch.long)), batch_size=BATCH, shuffle=False)

    # MIA AUC non-members pentru scenariul A
    if len(X_df_A) > 0:
        idx = np.random.choice(len(X_dr_A), size=len(X_df_A), replace=False)
        X_nm = X_dr_A[idx]; y_nm = y_dr_A[idx]
        mia_nonmembers_A = DataLoader(TensorDataset(torch.tensor(X_nm, dtype=torch.float32),
                                                    torch.tensor(y_nm, dtype=torch.long)), batch_size=BATCH, shuffle=False)
    else:
        mia_nonmembers_A = None

    # Scenariul B
    retain_loader_B = DataLoader(TensorDataset(torch.tensor(X_dr_B, dtype=torch.float32),
                                               torch.tensor(y_dr_B, dtype=torch.long)), batch_size=BATCH, shuffle=True)
    forget_loader_B = DataLoader(TensorDataset(torch.tensor(X_df_B, dtype=torch.float32),
                                               torch.tensor(y_df_B, dtype=torch.long)), batch_size=BATCH, shuffle=True)

    # 4) Oracle per scenariu (retrain fără Df), cu ES
    oracle_A, _, _ = train_teacher(X_train=X_dr_A, y_train=y_dr_A, X_test=X_te, y_test=y_te, tag="oracle_subsetA",
                                   patience=8, min_delta=0.0)
    oracle_B, _, _ = train_teacher(X_train=X_dr_B, y_train=y_dr_B, X_test=X_te, y_test=y_te, tag="oracle_fullclassB",
                                   patience=8, min_delta=0.0)

    results_rows = []

    # ---------- Scenario A: 2% subset forget ----------
    loaders_A = {"retain": retain_loader_A, "forget": forget_loader_A, "test": test_loader, "mia_nonmembers": mia_nonmembers_A}

    ga_A = method_gradient_ascent(teacher, forget_loader_A, epochs=25, tag="GA_A",
                                  eval_loader=test_loader, patience=6)
    evaluate_suite("GA", ga_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    fisher_A = method_fisher_unlearning(teacher, forget_loader_A, epochs=25, tag="Fisher_A",
                                        eval_loader=test_loader, patience=6)
    evaluate_suite("Fisher", fisher_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    neg_A = method_neggrad_plus(teacher, retain_loader_A, forget_loader_A, epochs=25, tag="NegGradPlus_A",
                                eval_loader=test_loader, patience=6)
    evaluate_suite("NegGradPlus", neg_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    cf1_A = method_cf_k(teacher, forget_loader_A, k=1, epochs=25, tag="CFk_A",
                        eval_loader=test_loader, patience=6)
    evaluate_suite("CF-k1", cf1_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    eu1_A = method_eu_k(teacher, retain_loader_A, k=1, epochs=25, tag="EUk_A",
                        eval_loader=test_loader, patience=6)
    evaluate_suite("EU-k1", eu1_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    scrub_A_simple = method_scrub(teacher, retain_loader_A, forget_loader_A,
                                  epochs=60, lam_r=1.0, lam_f=1.5, tag="SCRUB_A_simple",
                                  rewind=False, eval_loader=test_loader, patience=8)
    evaluate_suite("SCRUB", scrub_A_simple, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    scrub_A = method_scrub(teacher, retain_loader_A, forget_loader_A, epochs=60, lam_r=1.0, lam_f=1.5, tag="SCRUB_A",
                           rewind=True, eval_loader=test_loader, patience=8)
    evaluate_suite("SCRUB(+R)", scrub_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    cert_A = method_certified_unlearning(teacher, retain_loader_A,
                                         epochs=12, sigma=0.05, clip_norm=1.0, tag="Certified_A",
                                         eval_loader=test_loader, patience=6)
    evaluate_suite("Certified", cert_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    qmul_A = method_qmul(teacher, retain_loader_A, forget_loader_A,
                         epochs=12, alpha=1.0, tag="Q-MUL_A", eval_loader=test_loader, patience=6)
    evaluate_suite("Q-MUL", qmul_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    lca_A = method_label_complement_augmentation(teacher, retain_loader_A, forget_loader_A,
                                                 epochs=25, beta=1.0, tag="LCA_A",
                                                 eval_loader=test_loader, patience=6)
    evaluate_suite("LCA", lca_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    adv_A = method_adv_uniform(teacher, retain_loader_A, forget_loader_A, epochs=15, eps=0.1, lam=1.0, tag="ADVUNI_A",
                               eval_loader=test_loader, patience=6)
    evaluate_suite("ADV-UNIFORM", adv_A, teacher, oracle_A, loaders_A, "subset2pct", results_rows)

    # ---------- Scenario B: full-class forget ----------
    loaders_B = {"retain": retain_loader_B, "forget": forget_loader_B, "test": test_loader}

    ga_B = method_gradient_ascent(teacher, forget_loader_B, epochs=25, tag="GA_B",
                                  eval_loader=test_loader, patience=6)
    evaluate_suite("GA", ga_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    fisher_B = method_fisher_unlearning(teacher, forget_loader_B, epochs=25, tag="Fisher_B",
                                        eval_loader=test_loader, patience=6)
    evaluate_suite("Fisher", fisher_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    neg_B = method_neggrad_plus(teacher, retain_loader_B, forget_loader_B, epochs=25, tag="NegGradPlus_B",
                                eval_loader=test_loader, patience=6)
    evaluate_suite("NegGradPlus", neg_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    cf1_B = method_cf_k(teacher, forget_loader_B, k=1, epochs=25, tag="CFk_B",
                        eval_loader=test_loader, patience=6)
    evaluate_suite("CF-k1", cf1_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    eu1_B = method_eu_k(teacher, retain_loader_B, k=1, epochs=25, tag="EUk_B",
                        eval_loader=test_loader, patience=6)
    evaluate_suite("EU-k1", eu1_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    scrub_B_simple = method_scrub(teacher, retain_loader_B, forget_loader_B,
                                  epochs=60, lam_r=1.0, lam_f=1.5, tag="SCRUB_B_simple",
                                  rewind=False, eval_loader=test_loader, patience=8)
    evaluate_suite("SCRUB", scrub_B_simple, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    scrub_B = method_scrub(teacher, retain_loader_B, forget_loader_B, epochs=60, lam_r=1.0, lam_f=1.5, tag="SCRUB_B",
                           rewind=True, eval_loader=test_loader, patience=8)
    evaluate_suite("SCRUB(+R)", scrub_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    cert_B = method_certified_unlearning(teacher, retain_loader_B,
                                         epochs=12, sigma=0.05, clip_norm=1.0, tag="Certified_B",
                                         eval_loader=test_loader, patience=6)
    evaluate_suite("Certified", cert_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    qmul_B = method_qmul(teacher, retain_loader_B, forget_loader_B,
                         epochs=12, alpha=1.0, tag="Q-MUL_B", eval_loader=test_loader, patience=6)
    evaluate_suite("Q-MUL", qmul_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    lca_B = method_label_complement_augmentation(teacher, retain_loader_B, forget_loader_B,
                                                 epochs=25, beta=1.0, tag="LCA_B",
                                                 eval_loader=test_loader, patience=6)
    evaluate_suite("LCA", lca_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    adv_B = method_adv_uniform(teacher, retain_loader_B, forget_loader_B, epochs=15, eps=0.1, lam=1.0, tag="ADVUNI_B",
                               eval_loader=test_loader, patience=6)
    evaluate_suite("ADV-UNIFORM", adv_B, teacher, oracle_B, loaders_B, "fullclass", results_rows)

    # Save results CSV
    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(OUT_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved results CSV to: {csv_path}")

    # Simple plots
    try:
        import matplotlib.pyplot as plt
        for scenario in df["scenario"].unique():
            d = df[df["scenario"] == scenario]
            plt.figure(figsize=(10,5))
            for metric in ["acc_test_unlearn", "UQI", "Agree_test"]:
                if metric in d.columns:
                    plt.plot(d["method"], d[metric], marker="o", label=metric)
            plt.title(f"Scenario: {scenario}")
            plt.xticks(rotation=30)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(OUT_DIR, f"plot_{scenario}.png")
            plt.savefig(plot_path)
            print(f"Saved plot: {plot_path}")
    except Exception as e:
        print("Plotting failed:", e)

# ----------------------------
# Run
# ----------------------------
run_experiment()
