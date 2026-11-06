"""Run all experiments of the paper or partial:
    python run_experiments.py                 # everything
    python run_experiments.py complete er     # only these experiments
    python run_experiments.py --list          # print the runs without executing them
    python run_experiments.py -j 4            # 4 runs in parallel
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
AGGS = ['safe2d', 'geomed', 'mean', 'krum', 'trmean']
SEEDS = [0, 1, 2]
LOCAL = ['alie', 'minmax', 'ipm']


def settings(attacks, shares):
    return [('none', 0.0)] + [(a, b) for a in attacks for b in shares]


EXPERIMENTS = {
    'complete': [dict(dataset=d, graph='complete', placement='random', alpha=0.5, attacks=settings(LOCAL, s))
                 for d, s in [('digits', [0.2, 0.28, 0.3]), ('mnist1d', [0.2, 0.28, 0.3]), ('pathmnist', [0.2, 0.3])]],
    'regular': [dict(dataset=d, graph='regular', placement='safe', alpha=0.5, attacks=settings(LOCAL, [0.2, 0.28]))
                for d in ['digits', 'mnist1d', 'pathmnist']],
    'er': [dict(dataset=d, graph='er', placement='random', alpha=0.5,
                attacks=[('none', 0.0), ('spread', 0.3), ('ipm', 0.2), ('ipm', 0.3)])
           for d in ['digits', 'mnist1d', 'pathmnist']],
    'er_het': [dict(dataset=d, graph='er', placement='random', alpha=0.1, local_epochs=le,
                    attacks=settings(['alie', 'minmax'], [0.2, 0.3]))
               for d, le in [('digits', 5), ('mnist1d', 6)]],
}


def key(r): return (r['dataset'], r['attack'], float(r['beta']), r['agg'], int(r['seed']))


def runs(name):
    for e in EXPERIMENTS[name]:
        for attack, beta in e['attacks']:
            for agg in AGGS:
                for seed in SEEDS:
                    yield dict({k: v for k, v in e.items() if k != 'attacks'}, attack=attack, beta=beta, agg=agg, seed=seed)


def done(path):
    if not os.path.exists(path): return set()
    return {key(json.loads(l)) for l in open(path) if l.startswith('{')}


def execute(job, path):
    cmd = [sys.executable, os.path.join(HERE, 'safe2d_sim.py')] + \
          [x for k, v in job.items() for x in (f"--{k.replace('_', '-')}", str(v))]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith('{')), None)
    if line is None:
        print('FAILED', job, out.stderr[-500:], file=sys.stderr); return
    with open(path, 'a') as f: f.write(line + '\n')
    print('done', key(json.loads(line)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('experiments', nargs='*', help='subset of: ' + ', '.join(EXPERIMENTS))
    ap.add_argument('-j', '--jobs', type=int, default=2)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    for n in a.experiments:
        if n not in EXPERIMENTS: ap.error(f'unknown experiment {n}')
    a.experiments = a.experiments or list(EXPERIMENTS)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
    todo = []
    for name in a.experiments:
        path = os.path.join(HERE, 'results', f'{name}.jsonl')
        have = done(path); jobs = [j for j in runs(name) if key(j) not in have]
        print(f'{name}: {len(have)} done, {len(jobs)} to run')
        todo += [(j, path) for j in jobs]
    if a.list:
        for j, p in todo: print(os.path.basename(p), j)
    else:
        with ThreadPoolExecutor(a.jobs) as pool: list(pool.map(lambda jp: execute(*jp), todo))
