import numpy as np
from safe2d_code.utils import AGGS, AGG_NAMES, DATASETS, DATA_NAMES, cells

S = cells()
print(f'{len(S)} settings (experiment x dataset x attack x share)\n')
print(f"{'method':14s} {'avg f*':>7s} {'f*<0.3':>7s}")
for g in sorted(AGGS, key=lambda g: -np.mean([v[g] for v in S.values()])):
    print(f"{AGG_NAMES[g]:14s} {np.mean([v[g] for v in S.values()]):7.3f} {sum(v[g] < 0.3 for v in S.values()):7d}")

print('\nSafe2D head to head (difference > 0.01):')
for g in AGGS[1:]:
    better = sum(v['safe2d'] > v[g] + 0.01 for v in S.values()); worse = sum(v['safe2d'] < v[g] - 0.01 for v in S.values())
    print(f'  vs {AGG_NAMES[g]:13s} better in {better:2d}, worse in {worse:2d}, of {len(S)}')

print('\nAverage f* per dataset:')
for ds in DATASETS:
    V = [v for k, v in S.items() if k[1] == ds]
    print(f'  {DATA_NAMES[ds]:10s} ({len(V)} settings): ' + ', '.join(f'{AGG_NAMES[g]} {np.mean([v[g] for v in V]):.2f}' for g in AGGS))
