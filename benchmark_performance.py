#!/usr/bin/env python3
"""
Comprehensive performance benchmark for jakteristics optimizations.
Compares original vs optimized versions of compute_features and compute_scalars_stats.
"""

import time
import numpy as np
import psutil
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import multiprocessing
import gc

# Add current directory to path to import jakteristics
sys.path.insert(0, str(Path(__file__).parent))

import jakteristics
from jakteristics.constants import FEATURE_NAMES


class PerformanceBenchmark:
    def __init__(self):
        self.results = {}
        self.cpu_count = multiprocessing.cpu_count()
        
    def generate_test_data(self, n_points: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Generate synthetic 3D point cloud data for benchmarking."""
        np.random.seed(42)  # Reproducible results
        
        print(f"    Generating {n_points:,} point dataset...")
        
        if n_points >= 1000000:  # Optimized generation for large datasets
            # Generate base points efficiently
            points = np.random.randn(n_points, 3).astype(np.float64) * 10
            
            # Add fewer, larger clusters for large datasets
            n_clusters = min(100, n_points // 10000)  # Much fewer clusters
            cluster_size = n_points // n_clusters
            
            for i in range(n_clusters):
                start_idx = i * cluster_size
                end_idx = min((i + 1) * cluster_size, n_points)
                cluster_center = np.random.randn(3) * 50
                points[start_idx:end_idx] += cluster_center
                points[start_idx:end_idx] *= 0.5  # Tighter clusters
        else:
            # Original method for smaller datasets
            points = np.random.randn(n_points, 3) * 10
            
            # Add some clustered regions for realistic neighbor patterns
            n_clusters = max(1, n_points // 1000)
            for i in range(n_clusters):
                cluster_center = np.random.randn(3) * 50
                cluster_size = np.random.randint(50, 200)
                cluster_points = np.random.randn(cluster_size, 3) * 2 + cluster_center
                if i == 0:
                    points[:cluster_size] = cluster_points
                else:
                    start_idx = min(i * cluster_size, n_points - cluster_size)
                    points[start_idx:start_idx + cluster_size] = cluster_points
        
        # Generate scalar fields for scalars_stats testing
        scalar_fields = [
            np.random.randn(n_points),  # Height-like field
            np.random.exponential(2, n_points),  # Intensity-like field  
            np.random.uniform(0, 255, n_points),  # Color-like field
        ]
        
        points = points.astype(np.float64)
        print(f"    Dataset ready: {points.shape}")
        return points, scalar_fields

    def measure_cpu_utilization(self, duration: float = 0.1) -> float:
        """Measure average CPU utilization over a short period."""
        cpu_percentages = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            cpu_percentages.append(psutil.cpu_percent(interval=0.01, percpu=False))
            
        return np.mean(cpu_percentages) if cpu_percentages else 0.0

    def benchmark_compute_features(self, 
                                 points: np.ndarray, 
                                 search_radius: float,
                                 feature_names: List[str],
                                 num_threads: int) -> Dict[str, Any]:
        """Benchmark compute_features function."""
        
        # Warmup
        _ = jakteristics.compute_features(
            points[:100], search_radius, 
            feature_names=feature_names[:5], 
            num_threads=1
        )
        
        # Clear caches and collect garbage
        gc.collect()
        
        # Monitor CPU before
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Benchmark
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        features = jakteristics.compute_features(
            points, search_radius,
            feature_names=feature_names,
            num_threads=num_threads
        )
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        
        # Monitor CPU after
        final_cpu = psutil.cpu_percent(interval=0.1)
        
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_efficiency': cpu_time / (wall_time * num_threads) if wall_time > 0 else 0,
            'throughput_points_per_sec': len(points) / wall_time if wall_time > 0 else 0,
            'features_shape': features.shape,
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_utilization_change': final_cpu - initial_cpu,
        }

    def benchmark_compute_scalars_stats(self,
                                      points: np.ndarray,
                                      scalar_fields: List[np.ndarray],
                                      search_radius: float,
                                      num_threads: int) -> Dict[str, Any]:
        """Benchmark compute_scalars_stats function."""
        
        # Warmup
        _ = jakteristics.compute_scalars_stats(
            points[:100], search_radius,
            [field[:100] for field in scalar_fields[:1]],
            num_threads=1
        )
        
        # Clear caches and collect garbage
        gc.collect()
        
        # Monitor CPU before
        initial_cpu = psutil.cpu_percent(interval=0.1)
        
        # Benchmark
        start_time = time.perf_counter()
        start_cpu_time = time.process_time()
        
        stats = jakteristics.compute_scalars_stats(
            points, search_radius, scalar_fields,
            num_threads=num_threads
        )
        
        end_time = time.perf_counter()
        end_cpu_time = time.process_time()
        
        # Monitor CPU after
        final_cpu = psutil.cpu_percent(interval=0.1)
        
        wall_time = end_time - start_time
        cpu_time = end_cpu_time - start_cpu_time
        
        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_efficiency': cpu_time / (wall_time * num_threads) if wall_time > 0 else 0,
            'throughput_points_per_sec': len(points) / wall_time if wall_time > 0 else 0,
            'stats_shape': [s.shape for s in stats],
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_utilization_change': final_cpu - initial_cpu,
        }

    def run_comprehensive_benchmark(self, version_name: str) -> Dict[str, Any]:
        """Run comprehensive benchmarks across different scenarios."""
        
        print(f"\n=== Running {version_name} Benchmarks ===")
        
        scenarios = {
            'small': {'n_points': 1000, 'search_radius': 0.15},
            'medium': {'n_points': 10000, 'search_radius': 0.15}, 
            'large': {'n_points': 50000, 'search_radius': 0.15},
            'very_large': {'n_points': 10000000, 'search_radius': 0.15},
            'small_large_radius': {'n_points': 1000, 'search_radius': 0.3},
            'medium_large_radius': {'n_points': 10000, 'search_radius': 0.3},
        }
        
        thread_counts = [1, 2, 4, min(8, self.cpu_count)]
        if self.cpu_count > 8:
            thread_counts.append(self.cpu_count)
            
        feature_sets = {
            'basic': ['planarity', 'linearity', 'sphericity', 'verticality'],
            'geometric': ['eigenvalue_sum', 'omnivariance', 'anisotropy', 'planarity', 'linearity'],
            'all': FEATURE_NAMES
        }
        
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            print(f"\nScenario: {scenario_name} ({scenario['n_points']} points, r={scenario['search_radius']})")
            
            # Generate test data
            points, scalar_fields = self.generate_test_data(scenario['n_points'])
            
            results[scenario_name] = {}
            
            for thread_count in thread_counts:
                print(f"  Testing {thread_count} threads...")
                
                results[scenario_name][f'{thread_count}_threads'] = {}
                
                # Test compute_features with different feature sets
                # For very large datasets, only test basic features to save time
                test_feature_sets = feature_sets
                if scenario_name == 'very_large':
                    test_feature_sets = {'basic': feature_sets['basic']}
                
                for feature_set_name, features in test_feature_sets.items():
                    try:
                        result = self.benchmark_compute_features(
                            points, scenario['search_radius'], 
                            features, thread_count
                        )
                        results[scenario_name][f'{thread_count}_threads'][f'features_{feature_set_name}'] = result
                        
                        print(f"    Features ({feature_set_name}): {result['wall_time']:.3f}s, "
                              f"{result['throughput_points_per_sec']:.0f} pts/s, "
                              f"CPU eff: {result['cpu_efficiency']:.2f}")
                              
                    except Exception as e:
                        print(f"    Features ({feature_set_name}) failed: {e}")
                        results[scenario_name][f'{thread_count}_threads'][f'features_{feature_set_name}'] = {'error': str(e)}
                
                # Test compute_scalars_stats (skip for very large datasets)
                if scenario_name != 'very_large':
                    try:
                        result = self.benchmark_compute_scalars_stats(
                            points, scalar_fields, scenario['search_radius'], thread_count
                        )
                        results[scenario_name][f'{thread_count}_threads']['scalars_stats'] = result
                        
                        print(f"    Scalars stats: {result['wall_time']:.3f}s, "
                              f"{result['throughput_points_per_sec']:.0f} pts/s, "
                              f"CPU eff: {result['cpu_efficiency']:.2f}")
                              
                    except Exception as e:
                        print(f"    Scalars stats failed: {e}")
                        results[scenario_name][f'{thread_count}_threads']['scalars_stats'] = {'error': str(e)}
        
        return results

    def save_results(self, results: Dict[str, Any], version_name: str):
        """Save benchmark results to JSON file."""
        filename = f"benchmark_results_{version_name}.json"
        
        # Add system info
        results['system_info'] = {
            'cpu_count': self.cpu_count,
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'numpy_version': np.__version__,
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {filename}")

    def compare_results(self, original_file: str, optimized_file: str):
        """Compare original vs optimized results and generate report."""
        
        with open(original_file, 'r') as f:
            original = json.load(f)
        with open(optimized_file, 'r') as f:
            optimized = json.load(f)
            
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON REPORT")
        print("="*80)
        
        improvements = {}
        
        for scenario in original:
            if scenario == 'system_info':
                continue
                
            print(f"\n--- {scenario.upper()} ---")
            improvements[scenario] = {}
            
            for thread_config in original[scenario]:
                if thread_config not in optimized[scenario]:
                    continue
                    
                print(f"\n{thread_config}:")
                improvements[scenario][thread_config] = {}
                
                for test_name in original[scenario][thread_config]:
                    if test_name not in optimized[scenario][thread_config]:
                        continue
                        
                    orig = original[scenario][thread_config][test_name]
                    opt = optimized[scenario][thread_config][test_name]
                    
                    if 'wall_time' in orig and 'wall_time' in opt:
                        speedup = orig['wall_time'] / opt['wall_time']
                        cpu_eff_improvement = opt['cpu_efficiency'] - orig['cpu_efficiency']
                        throughput_improvement = opt['throughput_points_per_sec'] / orig['throughput_points_per_sec']
                        
                        improvements[scenario][thread_config][test_name] = {
                            'speedup': speedup,
                            'cpu_efficiency_improvement': cpu_eff_improvement,
                            'throughput_improvement': throughput_improvement
                        }
                        
                        print(f"  {test_name}:")
                        print(f"    Speedup: {speedup:.2f}x ({orig['wall_time']:.3f}s → {opt['wall_time']:.3f}s)")
                        print(f"    CPU Efficiency: {orig['cpu_efficiency']:.2f} → {opt['cpu_efficiency']:.2f} ({cpu_eff_improvement:+.2f})")
                        print(f"    Throughput: {throughput_improvement:.2f}x ({orig['throughput_points_per_sec']:.0f} → {opt['throughput_points_per_sec']:.0f} pts/s)")
        
        # Overall summary
        all_speedups = []
        all_cpu_improvements = []
        all_throughput_improvements = []
        
        for scenario in improvements:
            for thread_config in improvements[scenario]:
                for test in improvements[scenario][thread_config]:
                    data = improvements[scenario][thread_config][test]
                    all_speedups.append(data['speedup'])
                    all_cpu_improvements.append(data['cpu_efficiency_improvement'])
                    all_throughput_improvements.append(data['throughput_improvement'])
        
        if all_speedups:
            print(f"\n" + "="*80)
            print("OVERALL SUMMARY")
            print("="*80)
            print(f"Average Speedup: {np.mean(all_speedups):.2f}x (range: {np.min(all_speedups):.2f}x - {np.max(all_speedups):.2f}x)")
            print(f"Average CPU Efficiency Improvement: {np.mean(all_cpu_improvements):.3f}")
            print(f"Average Throughput Improvement: {np.mean(all_throughput_improvements):.2f}x")
            print(f"Best Speedup Achieved: {np.max(all_speedups):.2f}x")
            
        # Save comparison results
        comparison_results = {
            'improvements': improvements,
            'summary': {
                'avg_speedup': np.mean(all_speedups) if all_speedups else 0,
                'max_speedup': np.max(all_speedups) if all_speedups else 0,
                'avg_cpu_efficiency_improvement': np.mean(all_cpu_improvements) if all_cpu_improvements else 0,
                'avg_throughput_improvement': np.mean(all_throughput_improvements) if all_throughput_improvements else 0,
            }
        }
        
        with open('benchmark_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
            
        print(f"\nDetailed comparison saved to benchmark_comparison.json")


def main():
    """Main benchmark execution."""
    benchmark = PerformanceBenchmark()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        # Compare existing results
        benchmark.compare_results('benchmark_results_original.json', 'benchmark_results_current_ultra.json')
    else:
        # Run benchmarks
        version = sys.argv[1] if len(sys.argv) > 1 else 'test'
        
        try:
            results = benchmark.run_comprehensive_benchmark(version)
            benchmark.save_results(results, version)
            
            print(f"\nBenchmark completed successfully!")
            print(f"Run with 'python benchmark_performance.py compare' to compare original vs optimized results.")
            
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nBenchmark failed: {e}")
            raise


if __name__ == "__main__":
    main()