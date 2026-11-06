"""P2P federated learning simulator.
    python safe2d_sim.py --dataset digits --graph complete --attack ipm --beta 0.3 --agg safe2d --seed 0
"""
import argparse, json, os, time
import numpy as np
from scipy.optimize import linprog, minimize
from scipy.spatial import ConvexHull, HalfspaceIntersection

N_CLIENTS, DEGREE, BATCH = 64, 20, 8

DATASETS = {
    'digits':    dict(hidden=32, lr=0.1,  local_epochs=1, rounds=100),
    'mnist1d':   dict(hidden=64, lr=0.1,  local_epochs=3, rounds=120),
    'pathmnist': dict(hidden=32, lr=0.01, local_epochs=1, rounds=40),
}
AGGS = ['safe2d', 'geomed', 'mean', 'krum', 'trmean']
ATTACKS = ['none', 'ipm', 'alie', 'minmax', 'spread']
GRAPHS = ['er', 'regular', 'complete']

def load_dataset(name, data_dir):
    """Standardized (X_train, y_train, X_test, y_test)."""
    if name == 'digits':
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        X, y = load_digits(return_X_y=True)
        Xtr, Xte, ytr, yte = train_test_split(X / 16.0, y, test_size=0.25, stratify=y, random_state=0)
        return Xtr, ytr, Xte, yte
    if name == 'mnist1d':
        from mnist1d.data import make_dataset, get_dataset_args
        d = make_dataset(get_dataset_args())
        Xtr, ytr, Xte, yte = d['x'], d['y'], d['x_test'], d['y_test']
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        return (Xtr - mu) / sd, ytr, (Xte - mu) / sd, yte
    if name == 'pathmnist':
        d = np.load(os.path.join(data_dir, 'pathmnist.npz'))
        flat = lambda a: a.reshape(len(a), -1).astype(np.float32) / 255.0
        Xtr, Xte = flat(d['train_images']), flat(d['test_images'])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        return (Xtr - mu) / sd, d['train_labels'].ravel().astype(int), (Xte - mu) / sd, d['test_labels'].ravel().astype(int)
    raise ValueError(name)


def dirichlet_split(y, n_classes, alpha, rng):
    """Split sample indices among clients with a per-class Dirichlet(alpha) distribution."""
    idx = [[] for _ in range(N_CLIENTS)]
    for c in range(n_classes):
        ids = rng.permutation(np.where(y == c)[0])
        cuts = (np.cumsum(rng.dirichlet(alpha * np.ones(N_CLIENTS))) * len(ids)).astype(int)[:-1]
        for k, part in enumerate(np.split(ids, cuts)): idx[k] += part.tolist()
    for k in range(N_CLIENTS):  # every client gets at least a few samples
        if len(idx[k]) < 5: idx[k] += rng.choice(len(y), 5, replace=False).tolist()
    return [np.array(i) for i in idx]


# one-hidden-layer MLP, parameter vector flatten
class MLP:
    def __init__(self, in_dim, hidden, n_classes, lr, local_epochs):
        self.shapes = [(in_dim, hidden), (hidden,), (hidden, n_classes), (n_classes,)]
        off = np.cumsum([0] + [int(np.prod(s)) for s in self.shapes])
        self.slices = [slice(off[i], off[i + 1]) for i in range(4)]
        self.groups = [slice(off[0], off[2]), slice(off[2], off[4])]  # [W, b] of each layer, projected together
        self.in_dim, self.hidden, self.n_classes, self.lr, self.local_epochs = in_dim, hidden, n_classes, lr, local_epochs

    def init(self, rng):
        s = self.shapes
        return np.concatenate([rng.normal(0, np.sqrt(2 / self.in_dim), int(np.prod(s[0]))), np.zeros(self.hidden),
                               rng.normal(0, np.sqrt(1 / self.hidden), int(np.prod(s[2]))), np.zeros(self.n_classes)])

    def unpack(self, w): return [w[sl].reshape(sh) for sl, sh in zip(self.slices, self.shapes)]

    def accuracy(self, w, X, y):
        W1, b1, W2, b2 = self.unpack(w)
        return float(((np.maximum(X @ W1 + b1, 0) @ W2 + b2).argmax(1) == y).mean())

    def train(self, w, X, y, rng):
        """Local SGD (cross-entropy, batch size BATCH) for local_epochs epochs."""
        w = w.copy()
        for _ in range(self.local_epochs):
            perm = rng.permutation(len(y))
            for s in range(0, len(y), BATCH):
                b = perm[s:s + BATCH]; xb, yb = X[b], y[b]
                W1, b1, W2, b2 = self.unpack(w); h = np.maximum(xb @ W1 + b1, 0); z = h @ W2 + b2
                p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
                p[np.arange(len(yb)), yb] -= 1; p /= len(yb)
                dh = (p @ W2.T) * (h > 0)
                w -= self.lr * np.concatenate([(xb.T @ dh).ravel(), dh.sum(0), (h.T @ p).ravel(), p.sum(0)])
        return w

def make_graph(kind, seed):
    if kind == 'complete':
        return [np.r_[i, [j for j in range(N_CLIENTS) if j != i]] for i in range(N_CLIENTS)]
    if kind == 'regular':  # random DEGREE-regular graph
        import networkx as nx
        G = nx.random_regular_graph(DEGREE, N_CLIENTS, seed=seed + 1000)
        return [np.r_[i, sorted(G.neighbors(i))] for i in range(N_CLIENTS)]
    if kind == 'er':  # Erdos-Renyi with expected degree DEGREE
        rng = np.random.default_rng(seed + 1000)
        A = np.triu(rng.random((N_CLIENTS, N_CLIENTS)) < DEGREE / (N_CLIENTS - 1), 1); A = A | A.T
        return [np.r_[i, np.where(A[i])[0]] for i in range(N_CLIENTS)]
    raise ValueError(kind)


def _ok(peers, bad): 
    return all(sum(j in bad for j in peers[i]) < int(np.ceil(len(peers[i]) / 3)) for i in range(N_CLIENTS) if i not in bad)


def safe_placement(peers, nb, seed):
    """Byzantine set of size nb that keeps every honest peer set below one third (greedy, then swap search)."""
    rng, best = np.random.default_rng(seed + 777), set()
    for _ in range(400):
        bad = set()
        for v in rng.permutation(N_CLIENTS):
            if len(bad) == nb: break
            if _ok(peers, bad | {int(v)}): bad = bad | {int(v)}
        if len(bad) > len(best): best = bad
        if len(best) == nb: return best
    rng = np.random.default_rng(seed + 778)
    lim = np.array([int(np.ceil(len(p) / 3)) - 1 for p in peers])
    A = np.zeros((N_CLIENTS, N_CLIENTS), int)
    for i, p in enumerate(peers): A[i, p] = 1
    b = np.zeros(N_CLIENTS, bool); b[rng.choice(N_CLIENTS, nb, replace=False)] = True
    viol = lambda b: int(np.maximum(A @ b - lim, 0)[~b].sum())
    v = viol(b)
    for _ in range(50000):
        if v == 0: break
        i, j = rng.choice(np.where(b)[0]), rng.choice(np.where(~b)[0])
        b[i], b[j] = False, True; v2 = viol(b)
        if v2 <= v: v = v2
        else: b[i], b[j] = True, False
    if v: raise SystemExit(f'no safe placement of {nb} Byzantine clients on this graph')
    return set(np.where(b)[0].tolist())

_ANGLES = np.linspace(0, np.pi * 2, 180, endpoint=False)
DIRS = np.stack([np.cos(_ANGLES), np.sin(_ANGLES)], 1)  # 180 directions for the safe region

def safe_region(U):
    """Depth-ceil(n/3) Tukey region of the points U (n,2), outer approximation with 180 directions.
    Returns its Chebyshev centre and its vertices (a single vertex if the region is a point or empty)."""
    k = int(np.ceil(len(U) / 3)); kth = -np.sort(-(U @ DIRS.T), 0)[k - 1]  # k-th largest projection per direction
    r = linprog([0, 0, -1.0], A_ub=np.c_[DIRS, np.ones(len(DIRS))], b_ub=kth, bounds=[(None, None)] * 3, method='highs')
    if not r.success: q = U.mean(0); return q, q[None]
    q, radius = r.x[:2], r.x[2]
    if radius <= 1e-9 * (1 + np.abs(U).max()): return q, q[None]
    try:
        V = HalfspaceIntersection(np.c_[DIRS, -kth], q).intersections
        return q, V[ConvexHull(V).vertices]
    except Exception:
        return q, q[None]

def nearest_in_polygon(V, x):
    """Point of the convex polygon with vertices V closest to x."""
    if len(V) < 3: return V.mean(0)
    try: V = V[ConvexHull(V).vertices]
    except Exception: return V.mean(0)
    n, cross = len(V), lambda a, b: a[0] * b[1] - a[1] * b[0]
    if all(cross(V[(i + 1) % n] - V[i], x - V[i]) >= -1e-12 for i in range(n)): return x.copy()
    best, bd = None, np.inf
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]; t = np.clip(np.dot(x - a, b - a) / (np.dot(b - a, b - a) + 1e-300), 0, 1)
        p = a + t * (b - a); d = np.linalg.norm(p - x)
        if d < bd: best, bd = p, d
    return best


def min_norm_combination(P, q):
    """Convex weights mu (min ||mu||^2) with P^T mu = q, or None."""
    m = len(P)
    r = minimize(lambda l: np.sum(l * l), np.ones(m) / m, jac=lambda l: 2 * l,
                 constraints=[{'type': 'eq', 'fun': lambda l: P.T @ l - q, 'jac': lambda l: P.T},
                              {'type': 'eq', 'fun': lambda l: l.sum() - 1, 'jac': lambda l: np.ones((1, m))}],
                 bounds=[(0, 1)] * m, method='SLSQP', options={'maxiter': 300, 'ftol': 1e-12})
    if r.success and np.linalg.norm(P.T @ r.x - q) <= 1e-6 * (1 + np.abs(P).max()):
        x = np.clip(r.x, 0, None); return x / x.sum()
    import cvxpy as cp  # fallback solver
    l = cp.Variable(m)
    try: cp.Problem(cp.Minimize(cp.sum(cp.multiply(np.ones(m), cp.square(l)))), [P.T @ l == q, cp.sum(l) == 1, l >= 0]).solve(solver=cp.CLARABEL)
    except Exception: return None
    if l.value is None or np.linalg.norm(P.T @ l.value - q) > 1e-5: return None
    x = np.clip(l.value, 0, None); return x / x.sum()


def snapping_lift(U, q, V):
    """Weights over the peers: q = min-norm convex combination of the region vertices V and the own point U[0];
    each vertex passes its weight to the peer whose projection is nearest to it."""
    P = np.vstack([V, U[:1]])
    scale = np.median(np.linalg.norm(P - q, axis=1)) + 1e-12  # conditioning only; weights are scale-invariant
    mu = min_norm_combination((P - q) / scale, np.zeros(2))
    lam = np.zeros(len(U))
    if mu is None: lam[0] = 1.0; return lam  # keep own model
    np.add.at(lam, np.argmin(((V[:, None] - U[None]) ** 2).sum(-1), 1), mu[:-1]); lam[0] += mu[-1]
    return lam


def agg_safe2d(X, model, rng):
    """Safe2D for one client. X: peer models, row 0 = own model. Each layer gets its own private projection."""
    out = np.empty(X.shape[1])
    for sl in model.groups:
        Xl = X[:, sl]; Xl = Xl[first_unique(Xl)]                     # 1. deduplicate (own model stays first)
        U = Xl @ rng.normal(0, np.sqrt(0.5), size=(2, Xl.shape[1])).T  # 2. private random projection to the plane
        q, V = safe_region(U)                                         # 3. safe region
        if len(V) >= 3: q = nearest_in_polygon(V, U[0])               # 4. target: safe point nearest own projection
        out[sl] = snapping_lift(U, q, V) @ Xl                         # 5. lift back to a combination of peer models
    return out

def first_unique(X):
    """Indices of the first occurrence of each distinct row, in order."""
    seen, idx = set(), []
    for k, row in enumerate(X):
        h = row.tobytes()
        if h not in seen: seen.add(h); idx.append(k)
    return np.array(idx)


def agg_mean(X, f): return X.mean(0)


def agg_geomed(X, f):  # Weiszfeld
    z = X.mean(0)
    for _ in range(100):
        d = np.maximum(np.linalg.norm(X - z, axis=1), 1e-8); z2 = (X / d[:, None]).sum(0) / (1 / d).sum()
        if np.linalg.norm(z2 - z) < 1e-7: break
        z = z2
    return z


def agg_krum(X, f):
    n = len(X); m = max(n - max(int(round(f * n)), 0) - 2, 1)
    sq = (X * X).sum(1); D2 = np.maximum(sq[:, None] + sq[None] - 2 * X @ X.T, 0)
    return X[int(np.argmin(np.sort(D2, 1)[:, 1:m + 1].sum(1)))]


def agg_trmean(X, f):
    n = len(X); t = min(int(round(f * n)), (n - 1) // 2)
    return np.sort(X, 0)[t:n - t].mean(0)


BASELINES = {'mean': agg_mean, 'geomed': agg_geomed, 'krum': agg_krum, 'trmean': agg_trmean}


def attack_global(attack, H, nb, rng):
    """IPM and Spread: crafted from all honest models H (omniscient)."""
    mu = H.mean(0); spread = np.sqrt(((H - mu) ** 2).sum(1).max()) + 1e-12
    xi = rng.normal(size=(nb, H.shape[1])); xi /= np.linalg.norm(xi, axis=1, keepdims=True)
    if attack == 'ipm':     # -0.5 x honest mean, tiny independent noise (defeats exact deduplication)
        return -0.5 * mu + 1e-3 * spread * xi
    if attack == 'spread':  # consistent push away from the mean plus large independent noise
        u = -mu / (np.linalg.norm(mu) + 1e-12)
        return mu + 1.0 * spread * u + 30.0 * spread * xi
    raise ValueError(attack)


def attack_local(attack, Hn, nb):
    """ALIE and Min-Max: crafted by one Byzantine client from the honest models Hn it receives."""
    mu = Hn.mean(0)
    if attack == 'alie':
        from scipy.stats import norm
        s = int(np.ceil(N_CLIENTS / 2 + 1)) - nb
        return mu + norm.ppf(min(max((N_CLIENTS - s) / (N_CLIENTS - nb), 1e-12), 1 - 1e-12)) * Hn.std(0)
    if attack == 'minmax':  # largest step along -std that stays within the honest models' max pairwise distance
        p = -Hn.std(0); p /= np.linalg.norm(p) + 1e-12
        rmax = np.sqrt(((Hn[:, None] - Hn[None]) ** 2).sum(-1).max()) if len(Hn) > 1 else 0.0
        g, step, ok = 20.0, 10.0, 0.0
        for _ in range(200):
            if np.linalg.norm(Hn - (mu + g * p), axis=1).max() <= rmax + 1e-12: ok = g; g += step
            else: g -= step
            step /= 2
            if step < 1e-3: break
        return mu + ok * p
    raise ValueError(attack)


def run(dataset, graph, placement, attack, beta, agg, seed, alpha=0.5, local_epochs=None, rounds=None, data_dir='data'):
    cfg = DATASETS[dataset]
    local_epochs = local_epochs or cfg['local_epochs']; rounds = rounds or cfg['rounds']
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = load_dataset(dataset, data_dir)
    n_classes = int(ytr.max()) + 1
    parts = dirichlet_split(ytr, n_classes, alpha, np.random.default_rng(seed))
    model = MLP(Xtr.shape[1], cfg['hidden'], n_classes, cfg['lr'], local_epochs)
    peers = make_graph(graph, seed)

    nb = int(round(beta * N_CLIENTS)) if attack != 'none' else 0
    if nb and placement == 'safe': bad = safe_placement(peers, nb, seed)
    else: bad = set(rng.choice(N_CLIENTS, nb, replace=False).tolist()) if nb else set()
    good = [i for i in range(N_CLIENTS) if i not in bad]
    f = beta if beta > 0 else 0.1  # parameter of Krum and Trimmed-Mean

    W = np.repeat(model.init(rng)[None], N_CLIENTS, 0)
    curve = []
    for t in range(rounds):
        for i in good: W[i] = model.train(W[i], Xtr[parts[i]], ytr[parts[i]], rng)
        if attack in ('ipm', 'spread'):
            B = attack_global(attack, W[good], nb, rng)
            for j, b in enumerate(sorted(bad)): W[b] = B[j]
        elif nb:
            crafted = {b: attack_local(attack, W[[j for j in peers[b] if j not in bad] or good], nb) for b in bad}
            for b in bad: W[b] = crafted[b]
        newW, cache = W.copy(), {}
        for i in good:
            X = W[peers[i]]
            if agg == 'safe2d': newW[i] = agg_safe2d(X, model, rng)
            else:  # deterministic: clients with the same peer set share one computation
                key = tuple(sorted(int(j) for j in peers[i]))
                if key not in cache: cache[key] = BASELINES[agg](X[first_unique(X)], f)
                newW[i] = cache[key]
        W = newW
        if t % 5 == 4 or t == rounds - 1:
            accs = [model.accuracy(W[i], Xte, yte) for i in good]
            curve.append((t + 1, float(np.mean(accs)), float(np.min(accs))))
    return dict(dataset=dataset, graph=graph, placement=placement, alpha=alpha, local_epochs=local_epochs, rounds=rounds,
                attack=attack, beta=beta, agg=agg, seed=seed, final_mean=curve[-1][1], final_min=curve[-1][2],
                accs=[round(a, 4) for a in accs], curve=curve,
                max_local_byz=max(sum(j in bad for j in peers[i]) for i in good) if nb else 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', choices=DATASETS, default='digits')
    ap.add_argument('--graph', choices=GRAPHS, default='er')
    ap.add_argument('--placement', choices=['random', 'safe'], default='random',
                    help='safe: every honest client has fewer than a third Byzantine peers')
    ap.add_argument('--attack', choices=ATTACKS, default='none')
    ap.add_argument('--beta', type=float, default=0.0, help='share of Byzantine clients')
    ap.add_argument('--agg', choices=AGGS, default='safe2d')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--alpha', type=float, default=0.5, help='Dirichlet concentration of the data split')
    ap.add_argument('--local-epochs', type=int, help='default: per dataset')
    ap.add_argument('--rounds', type=int, help='default: per dataset')
    ap.add_argument('--data-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
    a = ap.parse_args()
    t0 = time.time()
    res = run(a.dataset, a.graph, a.placement, a.attack, a.beta, a.agg, a.seed, a.alpha, a.local_epochs, a.rounds, a.data_dir)
    res['secs'] = round(time.time() - t0, 1)
    print(json.dumps(res), flush=True)
