import collections, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
EXPERIMENTS = ['complete', 'regular', 'er', 'er_het']

CENTRAL = {'digits': 96.9, 'mnist1d': 58.4, 'pathmnist': 62.14}
CHANCE = {'digits': 10.0, 'mnist1d': 10.0, 'pathmnist': 100 / 9}

DATASETS = ['digits', 'mnist1d', 'pathmnist']
DATA_NAMES = {'digits': 'Digits', 'mnist1d': 'MNIST-1D', 'pathmnist': 'PathMNIST'}
AGGS = ['safe2d', 'geomed', 'trmean', 'mean', 'krum']
AGG_NAMES = {'safe2d': 'Safe2D', 'geomed': 'GeoMed', 'trmean': 'Trimmed-Mean', 'mean': 'Mean', 'krum': 'Krum'}
ATTACK_NAMES = {'none': 'No attack', 'alie': 'ALIE', 'minmax': 'Min-Max', 'ipm': 'IPM', 'spread': 'Spread'}


def fstar(dataset, acc):
    """Normalized accuracy: 0 = chance, 1 = centralized training. acc is a fraction (0-1)."""
    return (100 * np.asarray(acc) - CHANCE[dataset]) / (CENTRAL[dataset] - CHANCE[dataset])


def load(experiment):
    """Runs of one experiment, grouped by (dataset, attack, beta, agg)."""
    R = collections.defaultdict(list)
    for l in open(os.path.join(RESULTS, f'{experiment}.jsonl')):
        r = json.loads(l); R[(r['dataset'], r['attack'], r['beta'], r['agg'])].append(r)
    return R


def summary(runs):
    """(mean f* over honest clients, worst honest client f*), each averaged over seeds."""
    ds = runs[0]['dataset']
    v = np.array([(fstar(ds, r['accs']).mean(), fstar(ds, r['accs']).min()) for r in runs])
    return v.mean(0)


def cells():
    """All settings of the paper: {(experiment, dataset, attack, beta): {agg: mean f*}}."""
    out = collections.defaultdict(dict)
    for e in EXPERIMENTS:
        for (ds, at, b, agg), runs in load(e).items():
            out[(e, ds, at, b)][agg] = summary(runs)[0]
    return out
