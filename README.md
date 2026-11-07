## Safe Points versus Distances: An Analysis of Byzantine-Robust Aggregation for Peer-to-Peer Learning

Simulator, experiment runner and analysis scripts for the paper. All experiments are run on a Intel(R) Xeon 6248 workstation provided by [Research Computing department at UMass Boston](https://www.umb.edu/rc/).

### Files

| File | Purpose |
|---|---|
| `safe2d_sim.py` | P2P learning simulator: data, MLP, networks, attacks, Safe2D and the four baselines. |
| `run_experiments.py` | Defines all experiments and runs the missing ones into `results/`. |
| `central.py` | Centralized reference accuracy used to normalize f\*. |
| `ranking.py` | Overall experimental ranking and summary. |
| `results/` | Raw results of the paper, one JSON line per run. |
| `results.py` | Generates figures and summary to `./results`. |
| `utils.py` | Utilities functiong reading results and computing f\* (shared by the scripts below). |

### Setup

```
pip install -r requirements.txt
```
### Datasets
Digits ships from scikit-learn, MNIST-1D from the `mnist1d` package. PathMNIST is the 28×28 image set
`pathmnist.npz` downloaded from `medmnist` python package, placed in `/data` folder.

### Reproducing the result

```
python run_experiments.py -j 4     # all 1,050 runs; skips runs already in results/
python analysis.py                 # figures in figs/ and summary.txt
python ranking.py                  # numbers of Section 5.7
python central.py                  # centralized references (96.9%, 58.4%, 62.1%)
```

The results of the paper are included, so the analysis and reporting commands work without re-running the experiments.
A single run, for example Safe2D under IPM with 30% Byzantine clients on the complete network:

```
python safe2d_sim.py --dataset digits --graph complete --attack ipm --beta 0.3 --agg safe2d --seed 0
``` 