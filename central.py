"""Centralized accuracy for f*: same model and SGD trained on all training data, 3 seeds.

Digits and MNIST-1D: test accuracy after 100 epochs.
PathMNIST: test accuracy at the epoch with the best validation accuracy, over 40 epochs (its test accuracy fluctuates).

    python central.py digits mnist1d pathmnist
"""
import os, sys
import numpy as np
from safe2d_sim import DATASETS, MLP, load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def pathmnist_val():
    """Validation set, standardized with the training statistics. The mean is taken over raw pixels and then
    divided by 255, as in the computation of the paper's reference (62.14%); float32 rounding makes this differ
    slightly from load_dataset's order, which changes the selected epoch for seed 0 (61.92% otherwise)."""
    d = np.load(os.path.join(DATA_DIR, 'pathmnist.npz'))
    raw = d['train_images'].reshape(len(d['train_images']), -1).astype(np.float32)
    mu, sd = raw.mean(0) / 255, (raw / 255).std(0) + 1e-6
    return (d['val_images'].reshape(len(d['val_images']), -1).astype(np.float32) / 255 - mu) / sd, d['val_labels'].ravel()


def reference(dataset, seeds=(0, 1, 2)):
    cfg = DATASETS[dataset]
    Xtr, ytr, Xte, yte = load_dataset(dataset, DATA_DIR)
    val = pathmnist_val() if dataset == 'pathmnist' else None
    epochs = 40 if dataset == 'pathmnist' else 100
    accs = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        model = MLP(Xtr.shape[1], cfg['hidden'], int(ytr.max()) + 1, cfg['lr'], local_epochs=1)
        w, best = model.init(rng), (-1, None)
        for e in range(epochs):
            w = model.train(w, Xtr, ytr, rng)
            if val is not None:
                va = model.accuracy(w, *val)
                if va > best[0]: best = (va, model.accuracy(w, Xte, yte))
        accs.append(best[1] if val is not None else model.accuracy(w, Xte, yte))
        print(f'  {dataset} seed {seed}: {100 * accs[-1]:.2f}%', flush=True)
    return 100 * np.mean(accs)


if __name__ == '__main__':
    for ds in sys.argv[1:] or list(DATASETS):
        print(f'{ds}: centralized reference {reference(ds):.2f}%', flush=True)
