import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import (HERE, DATA_NAMES, ATTACK_NAMES, AGGS, AGG_NAMES, fstar, load, summary, cells)

OUT = os.path.join(HERE, 'results/figs')
SUMMARY_PATH = os.path.join(HERE, 'results/summary.txt')
COLOR = {'safe2d': '#2a78d6', 'geomed': '#eb6834', 'trmean': '#1baf7a', 'mean': '#eda100', 'krum': '#e87ba4'}
MARKER = {'safe2d': 'o', 'geomed': 's', 'trmean': '^', 'mean': 'D', 'krum': 'v'}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6})

COLS = ['geomed', 'safe2d', 'mean', 'krum', 'trmean']
ORDER = [('none', 0.0)] + [(a, b) for a in ['alie', 'minmax', 'ipm', 'spread'] for b in [0.2, 0.28, 0.3]]
ROW_ORDER_ER = [('none', 0.0), ('spread', 0.3), ('ipm', 0.2), ('ipm', 0.3)]


def _setting_label(attack, beta):
    return ATTACK_NAMES[attack] if attack == 'none' else f'{ATTACK_NAMES[attack]} {round(100 * beta)}%'


def _text_table(experiment, datasets, worst=True, order=ORDER):
    results = load(experiment)
    headers = ['Dataset', 'Setting'] + [AGG_NAMES[g] for g in COLS]
    rows = []

    for dataset in datasets:
        settings = [(attack, beta) for attack, beta in order
                    if (dataset, attack, beta, 'safe2d') in results]
        for attack, beta in settings:
            values = {agg: summary(results[(dataset, attack, beta, agg)]) for agg in COLS}
            best = max(round(value[0], 2) for value in values.values())
            cells = []
            for agg in COLS:
                mean, lowest = values[agg]
                cell = f'{mean:.2f}'
                if round(mean, 2) == best:
                    cell += '*'
                if worst:
                    cell += f' ({lowest:.2f})'
                cells.append(cell)
            rows.append([DATA_NAMES[dataset], _setting_label(attack, beta), *cells])

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        '  '.join(value.ljust(widths[i]) for i, value in enumerate(headers)),
        '  '.join('-' * width for width in widths),
    ]
    lines.extend('  '.join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows)
    return '\n'.join(lines)


def summary():
    sections = [
        ('Complete network - Digits and MNIST-1D', _text_table('complete', ['digits', 'mnist1d'])),
        ('Complete network - PathMNIST', _text_table('complete', ['pathmnist'])),
        ('Regular network', _text_table('regular', ['digits', 'mnist1d', 'pathmnist'])),
        ('Erdos-Renyi network', _text_table('er', ['digits', 'mnist1d', 'pathmnist'], worst=False,
                                            order=ROW_ORDER_ER)),
        ('Heterogeneous local epochs', _text_table('er_het', ['digits', 'mnist1d'])),
    ]
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as output:
        output.write('Safe2D experiment summary\n')
        output.write('Normalized accuracy f*; * marks the best mean f* in each setting.\n')
        output.write('For tables showing two values, parentheses contain the worst-client f*.\n\n')
        for title, table in sections:
            output.write(f'{title}\n')
            output.write(f'{table}\n\n')


def fig_ipm(C):
    """Figure 1: IPM on the complete network, f* versus the share of Byzantine clients."""
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.55), sharey=True)
    for ax, ds in zip(axs, ['digits', 'mnist1d', 'pathmnist']):
        shares = [0.0, 0.2, 0.28, 0.3] if ds != 'pathmnist' else [0.0, 0.2, 0.3]
        for agg in AGGS:
            ys = [summary(C[(ds, 'none', 0.0, agg) if b == 0 else (ds, 'ipm', b, agg)])[0] for b in shares]
            ax.plot(range(len(shares)), ys, color=COLOR[agg], marker=MARKER[agg], ms=4.5, lw=2.2 if agg == 'safe2d' else 1.4,
                    label=AGG_NAMES[agg], zorder=3 if agg == 'safe2d' else 2, markeredgecolor='white', markeredgewidth=0.6)
        ax.set_title(DATA_NAMES[ds], color=INK); ax.set_xlabel('Byzantine clients (%)')
        ax.set_xticks(range(len(shares))); ax.set_xticklabels([f'{round(100 * b)}' for b in shares])
        ax.grid(axis='y', color=GRID, lw=0.6); ax.set_ylim(-0.25, 1.05)
    axs[0].set_ylabel('normalized accuracy $f^*$')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=5, frameon=False, handlelength=1.8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=0.8, rect=(0, 0, 1, 0.9)); fig.savefig(os.path.join(OUT, 'fig_ipm.pdf')); plt.close(fig)


def fig_scatter(S):
    """Figure 2: Safe2D versus GeoMed in every setting."""
    fam = {'none': ('No attack', '#6b6a64', 'o'), 'alie': ('ALIE / Min-Max', '#1baf7a', 's'),
           'minmax': ('ALIE / Min-Max', '#1baf7a', 's'), 'ipm': ('IPM', '#2a78d6', '^'), 'spread': ('Spread', '#eb6834', 'D')}
    fig, ax = plt.subplots(figsize=(3.4, 3.6))
    ax.plot([-0.25, 1.05], [-0.25, 1.05], color=MUTED, lw=0.8, ls='--', zorder=1)
    ax.fill_between([-0.25, 1.05], [-0.25, 1.05], 1.05, color=COLOR['safe2d'], alpha=0.05, zorder=0)
    seen = set()
    for (e, ds, at, b), v in sorted(S.items()):
        lab, col, mk = fam[at]
        ax.scatter(v['geomed'], v['safe2d'], s=22, color=col, marker=mk, edgecolor='white', linewidth=0.5, zorder=3,
                   label=None if lab in seen else lab); seen.add(lab)
    ax.text(0.02, 0.97, 'Safe2D better', color=COLOR['safe2d'], fontsize=7.5, va='top')
    ax.text(1.0, -0.2, 'GeoMed better', color=COLOR['geomed'], fontsize=7.5, ha='right')
    ax.set_xlim(-0.25, 1.05); ax.set_ylim(-0.25, 1.05); ax.set_aspect('equal')
    ax.set_xlabel('GeoMed $f^*$'); ax.set_ylabel('Safe2D $f^*$'); ax.grid(color=GRID, lw=0.6)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), frameon=False, handletextpad=0.3, ncol=4, columnspacing=0.9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_scatter.pdf')); plt.close(fig)


def fig_drift(C):
    """Figure 3: accuracy over rounds, IPM 30% on the complete network, Safe2D and GeoMed."""
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2), sharey=True)
    for ax, ds in zip(axs, ['digits', 'mnist1d']):
        for agg in ['safe2d', 'geomed']:
            runs = C[(ds, 'ipm', 0.3, agg)]
            t = [c[0] for c in runs[0]['curve']]
            mean = np.mean([[fstar(ds, c[1]) for c in r['curve']] for r in runs], 0)
            worst = np.mean([[fstar(ds, c[2]) for c in r['curve']] for r in runs], 0)
            ax.plot(t, mean, color=COLOR[agg], lw=2.0, label=f'{AGG_NAMES[agg]}, mean client')
            ax.plot(t, worst, color=COLOR[agg], lw=1.2, ls=(0, (3, 2)), label=f'{AGG_NAMES[agg]}, worst client')
        ax.axhline(0, color=MUTED, lw=0.6); ax.set_title(f'{DATA_NAMES[ds]}, IPM 30%', color=INK)
        ax.set_xlabel('round'); ax.grid(axis='y', color=GRID, lw=0.6)
    axs[0].set_ylabel('normalized accuracy $f^*$'); axs[0].set_ylim(-0.35, 1.0)
    axs[1].legend(loc='upper right', frameon=False, ncol=1)
    fig.tight_layout(w_pad=0.8); fig.savefig(os.path.join(OUT, 'fig_drift.pdf')); plt.close(fig)


def fig_rank(S):
    """Figure 4: average f* over all settings and number of settings below 0.3."""
    n = len(S)
    avg = {g: np.mean([v[g] for v in S.values()]) for g in AGGS}
    low = {g: sum(v[g] < 0.3 for v in S.values()) for g in AGGS}
    order = sorted(AGGS, key=lambda g: (-round(avg[g], 2), low[g]))
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 1.9))
    for ax, d, xl, fmt in [(axs[0], avg, f'average $f^*$ over {n} settings', '{:.2f}'),
                           (axs[1], low, f'settings with $f^*<0.3$ (of {n})', '{:d}')]:
        y = np.arange(len(order))[::-1]
        ax.barh(y, [d[g] for g in order], color=[COLOR[g] for g in order], height=0.62)
        for yi, g in zip(y, order): ax.text(d[g] + max(d.values()) * 0.02, yi, fmt.format(d[g]), va='center', color=INK, fontsize=7.5)
        ax.set_yticks(y); ax.set_yticklabels([AGG_NAMES[g] for g in order]); ax.set_xlabel(xl)
        ax.set_xlim(0, max(d.values()) * 1.15); ax.grid(axis='x', color=GRID, lw=0.6); ax.tick_params(axis='y', length=0)
    fig.tight_layout(w_pad=1.5); fig.savefig(os.path.join(OUT, 'fig_rank.pdf')); plt.close(fig)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    C, S = load('complete'), cells()
    fig_ipm(C); fig_scatter(S); fig_drift(C); fig_rank(S)
    summary()
    print(f'Added 4 figures to ./figs/ ({len(S)} settings)')
    print('Added ./summary.txt')
