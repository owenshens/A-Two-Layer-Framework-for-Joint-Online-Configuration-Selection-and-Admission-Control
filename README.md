# Alibaba SP-UCB-OLP Experiment

Self-contained experiment for evaluating SP-UCB-OLP on real Alibaba Cluster Trace data.

## Overview

This experiment tests online learning algorithms for resource allocation using the Alibaba Cluster Trace 2018 dataset.

**Key Features:**
- Real trace data (no synthetic generation)
- Original temporal order (no shuffling)
- New reward formula based on task duration and resource consumption
- K=3 regimes (8-bit, 4-bit quantization, batching)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download real Alibaba data (~130MB download, ~800MB extracted)
python download_data.py

# 3. Run smoke test
python run_experiment.py --smoke

# 4. Run full experiment
python run_experiment.py

# 5. Visualize results
python visualize_results.py
```

## Data Download Options

```bash
# Download full dataset (~13.4M rows, ~800MB)
python download_data.py

# Download and limit to first 100K rows (faster for testing)
python download_data.py --max-rows 100000

# Create synthetic data for quick testing (10K rows)
python download_data.py --synthetic

# Force re-download
python download_data.py --force
```

## Reward Formula

The reward for each arrival under regime θ is:

```
r = c1[θ] × cpu + c2[θ] × mem + noise
```

Where:
- `cpu`: Normalized CPU consumption (plan_cpu / 100, in [0,1])
- `mem`: Normalized memory consumption (plan_mem / 100, in [0,1])
- `c1[θ], c2[θ]`: Regime-specific reward coefficients
- `noise`: Gaussian(0, σ=0.1) additive noise

This creates a **stationary LP structure** where different regimes have different reward-per-consumption ratios. The algorithm must learn which regime provides the best efficiency.

The consumption vector is `a = [cpu, mem]`.

## Regime Configuration (K=3)

| Regime | Name | c1 (CPU coef) | c2 (Mem coef) | Description |
|--------|------|---------------|---------------|-------------|
| 0 | CPU-heavy | 2.0 | 0.5 | Better for CPU-intensive tasks |
| 1 | Memory-heavy | 0.5 | 2.0 | Better for memory-intensive tasks |
| 2 | Balanced | 1.2 | 1.2 | Balanced reward structure |

The optimal regime depends on the resource profile of arrivals:
- If avg(cpu) > avg(mem): Regime 0 (CPU-heavy) tends to win
- If avg(mem) > avg(cpu): Regime 1 (Memory-heavy) tends to win
- If balanced: Regime 2 may be competitive

## Algorithms

1. **SP-UCB-OLP** (α=0.1): Main algorithm with exploration
2. **Greedy** (α=0): Pure exploitation baseline
3. **Random**: Uniform random selection
4. **Oracle**: Perfect information upper bound

## Experiment Configuration

**Full Config:**
- T = 10,000 arrivals
- ρ values: [0.5, 0.7, 1.0, 1.2]
- Seeds: [42, 43, 44, 45, 46]
- Total runs: 80 (4 algorithms × 4 ρ × 5 seeds)

**Smoke Test:**
- T = 1,000
- ρ values: [0.7, 1.0]
- Seeds: [42]

## File Structure

```
alibaba_experiment/
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── download_data.py            # Data downloader
├── run_experiment.py           # Main experiment runner
├── visualize_results.py        # Visualization script
├── algorithms/
│   ├── __init__.py
│   ├── base.py                 # Base algorithm class
│   ├── sp_ucb_olp.py          # SP-UCB-OLP implementation
│   └── baselines.py           # Greedy, Random, Oracle
├── data/
│   ├── __init__.py
│   ├── alibaba_loader.py      # Data loader
│   └── raw/                   # Downloaded trace data
│       └── batch_task.csv
└── results/
    ├── alibaba_results_*.csv  # Raw results
    ├── alibaba_summary_*.csv  # Summary statistics
    └── figures/               # Plots
```

## Data Source

**Alibaba Cluster Trace 2018:**
- Repository: https://github.com/alibaba/clusterdata
- Direct download: http://clusterdata2018pubcn.oss-cn-beijing.aliyuncs.com/batch_task.tar.gz
- Path: cluster-trace-v2018/

**Dataset Statistics (Full):**
| Metric | Value |
|--------|-------|
| Total rows | 13,442,065 |
| Time span | 8.9 days |
| CPU mean | 83.6 (0-100 scale) |
| Memory mean | 34.9 (0-100 scale) |
| Duration mean | 96.1 seconds |
| File size | ~800 MB |

**Columns:**
- `task_name`: Task identifier
- `instance_num`: Number of instances
- `job_name`: Job identifier
- `task_type`: Task type (map/reduce/etc.)
- `status`: Task status (Terminated, Failed, etc.)
- `start_time`: Start timestamp (seconds)
- `end_time`: End timestamp (seconds)
- `plan_cpu`: Planned CPU (0-100 scale)
- `plan_mem`: Planned memory (0-100 scale)

## Usage

### Download Data

```bash
# Automatic download (may require manual intervention)
python download_data.py

# Generate synthetic data for testing
python download_data.py --synthetic
```

### Run Experiments

```bash
# Smoke test (quick)
python run_experiment.py --smoke

# Full experiment
python run_experiment.py

# Custom configuration
python run_experiment.py --rho 0.5,0.7 --seeds 42,43 --T 5000
```

### Visualize Results

```bash
# Auto-detect latest results
python visualize_results.py

# Specify results file
python visualize_results.py --results ./results/alibaba_results_20240101_120000.csv
```

## Expected Output

### Competitive Ratio
- **Oracle**: ~0.98-1.00 (upper bound)
- **SP-UCB-OLP**: ~0.60-0.85 (depends on ρ)
- **Greedy**: ~0.55-0.80
- **Random**: ~0.30-0.50

### Generated Files
- `results/alibaba_results_*.csv`: Per-run results
- `results/alibaba_summary_*.csv`: Aggregated statistics
- `results/figures/alibaba_boxplots.png`: Boxplots by algorithm and ρ
- `results/figures/alibaba_performance_vs_rho.png`: Performance curves

## Notes

1. **Original Order**: Data is sorted by `start_time` and processed sequentially to mimic real operations.

2. **No Shuffling**: Unlike previous experiments, we do NOT shuffle the trace data. This preserves temporal patterns in the workload.

3. **Noise**: LogNormal noise is applied with seed-dependent randomness for reproducibility.

4. **Budget**: Nominal budget is set to allow ~50% acceptance rate on average.

## Citation

If you use this code or the Alibaba trace data, please cite:

```bibtex
@inproceedings{alibaba2018trace,
  title={Alibaba Cluster Trace Program},
  author={Alibaba Group},
  year={2018},
  url={https://github.com/alibaba/clusterdata}
}
```
