"""
Alibaba Trace Data Loader

Loads Alibaba Cluster Trace 2018 data with:
- Reward formula: r = c1[θ] * cpu + c2[θ] * mem + noise
- Original temporal order (no shuffling)
- K=3 regimes with different reward coefficients

Resources (d=2):
- Resource 0: CPU
- Resource 1: Memory

The reward is directly proportional to consumption with regime-specific
coefficients. This creates a stationary LP structure where the algorithm
must learn which regime has the best reward-per-consumption ratio.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, Any


# Regime configurations for K=3
# Reward: r = c1 * cpu + c2 * mem + noise
# Different regimes have different reward-per-resource ratios
REGIME_CONFIGS = {
    0: {
        'name': 'CPU-heavy',
        'c1': 2.0,   # High reward per CPU
        'c2': 0.5,   # Low reward per memory
        'cpu_mult': 1.0,
        'mem_mult': 1.0,
        'description': 'Better for CPU-intensive tasks',
    },
    1: {
        'name': 'Memory-heavy',
        'c1': 0.5,   # Low reward per CPU
        'c2': 2.0,   # High reward per memory
        'cpu_mult': 1.0,
        'mem_mult': 1.0,
        'description': 'Better for memory-intensive tasks',
    },
    2: {
        'name': 'Balanced',
        'c1': 1.2,   # Moderate reward per CPU
        'c2': 1.2,   # Moderate reward per memory
        'cpu_mult': 1.0,
        'mem_mult': 1.0,
        'description': 'Balanced reward structure',
    },
}


class AlibabaTraceLoader:
    """
    Data loader for Alibaba Cluster Trace experiments.

    Reward Formula:
        r = c1[θ] × cpu + c2[θ] × mem + noise

    where:
        - cpu: plan_cpu / 100 (normalized to [0,1])
        - mem: plan_mem / 100 (normalized to [0,1])
        - c1[θ], c2[θ]: regime-specific reward coefficients
        - noise: Gaussian(0, σ)

    This creates a stationary LP structure where different regimes
    have different reward-per-consumption ratios.

    Consumption:
        a = [cpu, mem]

    Parameters
    ----------
    data_path : str or Path
        Path to batch_task.csv
    T : int
        Time horizon (number of arrivals to use)
    seed : int
        Random seed for noise generation
    noise_sigma : float
        Standard deviation of additive Gaussian noise (default: 0.1)
    """

    def __init__(
        self,
        data_path: str,
        T: int,
        seed: int = 42,
        noise_sigma: float = 0.1,
    ):
        self.data_path = Path(data_path)
        self.T = T
        self.seed = seed
        self.noise_sigma = noise_sigma

        # Fixed dimensions
        self.K = 3  # Number of regimes
        self.d = 2  # CPU, Memory

        # Initialize RNG
        self._rng = np.random.RandomState(seed)

        # Load and process data
        self._load_data()
        self._compute_arrivals()
        self._compute_budget()

    def _load_data(self):
        """Load raw CSV data and sort by timestamp."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load CSV
        df = pd.read_csv(self.data_path)

        # Validate required columns
        required = ['plan_cpu', 'plan_mem', 'start_time', 'end_time']
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Sort by start_time to preserve original temporal order
        df = df.sort_values('start_time').reset_index(drop=True)

        # Filter out invalid entries
        df['duration'] = df['end_time'] - df['start_time']
        df = df[df['duration'] > 0]
        df = df[df['plan_cpu'] > 0]
        df = df[df['plan_mem'] > 0]

        # Limit to T samples
        if len(df) < self.T:
            print(f"WARNING: Only {len(df)} valid samples, requested T={self.T}")
            self.T = len(df)

        self._df = df.iloc[:self.T].reset_index(drop=True)

        print(f"Loaded {len(self._df)} arrivals from {self.data_path}")

    def _compute_arrivals(self):
        """Compute arrivals with regime-specific reward coefficients."""
        df = self._df

        # Extract base consumption values (normalized to [0,1])
        cpu_base = (df['plan_cpu'] / 100.0).values.clip(0.01, 1.0)
        mem_base = (df['plan_mem'] / 100.0).values.clip(0.01, 1.0)

        # Generate additive Gaussian noise (one per arrival, shared across regimes)
        noise = self._rng.normal(0, self.noise_sigma, size=self.T)

        # Store arrivals for each regime
        self._arrivals = {}

        for theta in range(self.K):
            config = REGIME_CONFIGS[theta]
            c1 = config['c1']  # Reward per CPU
            c2 = config['c2']  # Reward per memory

            # Compute regime-specific reward: r = c1 * cpu + c2 * mem + noise
            r_theta = c1 * cpu_base + c2 * mem_base + noise
            r_theta = np.maximum(r_theta, 0.01)  # Ensure positive rewards

            # Consumption (same for all regimes in this formulation)
            cpu_theta = cpu_base * config['cpu_mult']
            mem_theta = mem_base * config['mem_mult']

            # Store as (T, d+1) array: [reward, cpu, mem]
            arrivals = np.zeros((self.T, self.d + 1))
            arrivals[:, 0] = r_theta
            arrivals[:, 1] = cpu_theta
            arrivals[:, 2] = mem_theta

            self._arrivals[theta] = arrivals

        # Store base values for statistics
        self._cpu_base = cpu_base
        self._mem_base = mem_base

    def _compute_budget(self):
        """Compute nominal budget based on average consumption."""
        # Average consumption across all regimes
        total_cpu = sum(np.mean(self._arrivals[theta][:, 1]) for theta in range(self.K))
        total_mem = sum(np.mean(self._arrivals[theta][:, 2]) for theta in range(self.K))

        avg_cpu = total_cpu / self.K
        avg_mem = total_mem / self.K

        # Nominal budget: enough to accept ~50% of arrivals
        self._nominal_budget = 0.5 * self.T * np.array([avg_cpu, avg_mem])

    def get_arrival(self, theta: int, t: int) -> Tuple[float, np.ndarray]:
        """
        Get arrival at timestep t under configuration theta.

        Parameters
        ----------
        theta : int
            Configuration index in [0, K-1]
        t : int
            Timestep in [0, T-1]

        Returns
        -------
        r : float
            Reward value
        a : np.ndarray
            Resource consumption vector (d,)
        """
        if theta < 0 or theta >= self.K:
            raise ValueError(f"theta {theta} out of range [0, {self.K-1}]")
        if t < 0 or t >= self.T:
            raise ValueError(f"t {t} out of range [0, {self.T-1}]")

        arrival = self._arrivals[theta][t]
        return float(arrival[0]), arrival[1:].copy()

    def get_budget(self, rho: float) -> np.ndarray:
        """
        Get total budget scaled by rho.

        Parameters
        ----------
        rho : float
            Budget scaling factor (0.5 = tight, 1.0 = nominal, 1.5 = loose)

        Returns
        -------
        B : np.ndarray
            Total budget vector (d,)
        """
        return rho * self._nominal_budget

    def get_all_samples(self, theta: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get all samples for a configuration.

        Returns
        -------
        rewards : np.ndarray
            All rewards (T,)
        consumptions : np.ndarray
            All consumptions (T, d)
        """
        arrivals = self._arrivals[theta]
        return arrivals[:, 0], arrivals[:, 1:]

    def get_regime_name(self, theta: int) -> str:
        """Get human-readable regime name."""
        return REGIME_CONFIGS.get(theta, {}).get('name', f'Regime {theta}')

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the loaded data."""
        stats = {
            'T': self.T,
            'K': self.K,
            'd': self.d,
            'data_path': str(self.data_path),
            'seed': self.seed,
            'noise_sigma': self.noise_sigma,
            'regimes': {},
        }

        for theta in range(self.K):
            config = REGIME_CONFIGS[theta]
            rewards, consumptions = self.get_all_samples(theta)
            stats['regimes'][theta] = {
                'name': self.get_regime_name(theta),
                'c1': config['c1'],
                'c2': config['c2'],
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'mean_cpu': float(np.mean(consumptions[:, 0])),
                'mean_mem': float(np.mean(consumptions[:, 1])),
                'efficiency': float(np.mean(rewards) / (np.mean(consumptions[:, 0]) + np.mean(consumptions[:, 1]))),
            }

        return stats


def test_loader():
    """Test the Alibaba trace loader."""
    import sys

    print("=" * 60)
    print("Testing Alibaba Trace Loader")
    print("=" * 60)

    # Try to find data file
    data_paths = [
        Path(__file__).parent / 'raw' / 'batch_task.csv',
        Path('./data/raw/batch_task.csv'),
    ]

    data_path = None
    for p in data_paths:
        if p.exists():
            data_path = p
            break

    if data_path is None:
        print("ERROR: Data file not found. Run download_data.py first.")
        sys.exit(1)

    # Create loader
    loader = AlibabaTraceLoader(
        data_path=data_path,
        T=1000,
        seed=42
    )

    # Print statistics
    stats = loader.get_statistics()
    print(f"\nLoaded data from: {stats['data_path']}")
    print(f"T={stats['T']}, K={stats['K']}, d={stats['d']}")
    print(f"Noise sigma: {stats['noise_sigma']}")

    print("\nRegime Statistics:")
    for theta, regime_stats in stats['regimes'].items():
        print(f"\n  Regime {theta}: {regime_stats['name']}")
        print(f"    Mean reward: {regime_stats['mean_reward']:.2f}")
        print(f"    Mean CPU: {regime_stats['mean_cpu']:.4f}")
        print(f"    Mean Memory: {regime_stats['mean_mem']:.4f}")
        print(f"    Efficiency: {regime_stats['efficiency']:.2f}")

    print(f"\nNominal Budget (rho=1.0): {loader.get_budget(1.0)}")

    # Test get_arrival
    print("\nSample arrivals (t=0):")
    for theta in range(loader.K):
        r, a = loader.get_arrival(theta, 0)
        print(f"  Regime {theta}: r={r:.2f}, a=[{a[0]:.4f}, {a[1]:.4f}]")

    print("\n" + "=" * 60)
    print("Test passed!")


if __name__ == "__main__":
    test_loader()
