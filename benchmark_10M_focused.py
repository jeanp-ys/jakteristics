#!/usr/bin/env python3
"""
Focused 10M point benchmark comparing master vs ultra-optimized performance.
"""

import time
import numpy as np
import psutil
import json
import sys
import multiprocessing
import gc
from pathlib import Path

# Add current directory to path to import jakteristics
sys.path.insert(0, str(Path(__file__).parent))

import jakteristics

def generate_10M_dataset():
    """Generate 10M point dataset efficiently."""
    print("Generating 10M point dataset...")
    np.random.seed(42)
    
    # Generate clustered data efficiently
    n_points = 10_000_000
    n_clusters = 1000
    cluster_size = n_points // n_clusters
    
    points = np.zeros((n_points, 3), dtype=np.float64)
    
    # Generate cluster centers
    cluster_centers = np.random.randn(n_clusters, 3) * 100
    
    # Assign points to clusters
    for i in range(n_clusters):
        start_idx = i * cluster_size
        end_idx = min((i + 1) * cluster_size, n_points)
        
        # Generate points around cluster center
        cluster_points = np.random.randn(end_idx - start_idx, 3) * 5
        cluster_points += cluster_centers[i]
        points[start_idx:end_idx] = cluster_points
    
    print(f"Generated {n_points:,} points")
    return points

def benchmark_scenario(points, search_radius, num_threads, features):
    """Benchmark a specific scenario."""
    print(f"  {num_threads} threads: ", end="", flush=True)
    
    # Memory measurement
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    # Benchmark
    start_time = time.perf_counter()
    cpu_start = time.process_time()
    
    try:
        result = jakteristics.compute_features(
            points, 
            search_radius, 
            feature_names=features,
            num_threads=num_threads
        )
        
        end_time = time.perf_counter()
        cpu_end = time.process_time()
        
        wall_time = end_time - start_time
        cpu_time = cpu_end - cpu_start
        cpu_efficiency = cpu_time / (wall_time * num_threads) if wall_time > 0 else 0
        throughput = len(points) / wall_time if wall_time > 0 else 0
        
        mem_after = process.memory_info().rss / 1024 / 1024
        
        print(f"{wall_time:.1f}s, {throughput:,.0f} pts/s, {cpu_efficiency:.2f} eff, {mem_after:.0f}MB")
        
        return {
            'wall_time': wall_time,
            'cpu_time': cpu_time,
            'cpu_efficiency': cpu_efficiency,
            'throughput_points_per_sec': throughput,
            'features_shape': list(result.shape),
            'memory_usage_mb': mem_after,
            'memory_delta_mb': mem_after - mem_before,
            'success': True
        }
        
    except Exception as e:
        print(f"FAILED: {e}")
        return {'error': str(e), 'success': False}

def run_10M_benchmark(version_name):
    """Run focused 10M point benchmark."""
    print(f"\n=== 10M Point Benchmark ({version_name}) ===")
    
    # Generate dataset once
    points = generate_10M_dataset()
    
    # Test configurations
    features = ['planarity', 'linearity', 'sphericity', 'verticality']  # Basic set
    search_radius = 2.0  # Medium radius for realistic neighbors
    
    # Focus on realistic thread counts
    thread_counts = [1, 2, 4, 8, 16, 32]
    
    results = {}
    results['10M_basic'] = {}
    
    print(f"\nTesting 10M points with {len(features)} features, radius={search_radius}")
    
    for thread_count in thread_counts:
        result = benchmark_scenario(points, search_radius, thread_count, features)
        results['10M_basic'][f'{thread_count}_threads'] = {'features_basic': result}
        
        # Force garbage collection
        gc.collect()
    
    # Add system info
    results['system_info'] = {
        'cpu_count': multiprocessing.cpu_count(),
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'dataset_size': len(points),
        'features_tested': features,
        'search_radius': search_radius
    }
    
    # Save results
    filename = f"benchmark_10M_{version_name}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    return results

def compare_10M_results(master_file, optimized_file):
    """Compare 10M benchmark results."""
    
    try:
        with open(master_file, 'r') as f:
            master = json.load(f)
        with open(optimized_file, 'r') as f:
            optimized = json.load(f)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return
    
    print("\n" + "="*80)
    print("10M POINT PERFORMANCE COMPARISON")
    print("="*80)
    
    scenario = '10M_basic'
    if scenario not in master or scenario not in optimized:
        print("Missing benchmark data")
        return
    
    all_speedups = []
    
    print(f"\n10M Points Basic Features Comparison:")
    print(f"{'Threads':<8} {'Master (s)':<12} {'Optimized (s)':<15} {'Speedup':<10} {'Throughput Gain':<15}")
    print("-" * 70)
    
    for thread_config in sorted(master[scenario].keys(), key=lambda x: int(x.split('_')[0])):
        if thread_config not in optimized[scenario]:
            continue
            
        master_result = master[scenario][thread_config]['features_basic']
        opt_result = optimized[scenario][thread_config]['features_basic']
        
        if not (master_result.get('success') and opt_result.get('success')):
            print(f"{thread_config.replace('_threads', ''):<8} {'FAILED':<12}")
            continue
        
        speedup = master_result['wall_time'] / opt_result['wall_time']
        throughput_ratio = opt_result['throughput_points_per_sec'] / master_result['throughput_points_per_sec']
        
        all_speedups.append(speedup)
        
        threads = thread_config.replace('_threads', '')
        print(f"{threads:<8} {master_result['wall_time']:<12.1f} {opt_result['wall_time']:<15.1f} "
              f"{speedup:<10.2f}x {throughput_ratio:<15.2f}x")
    
    if all_speedups:
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Average Speedup: {np.mean(all_speedups):.2f}x")
        print(f"Best Speedup: {np.max(all_speedups):.2f}x") 
        print(f"Worst Speedup: {np.min(all_speedups):.2f}x")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        compare_10M_results('benchmark_10M_master.json', 'benchmark_10M_optimized.json')
    else:
        version = sys.argv[1] if len(sys.argv) > 1 else 'test'
        run_10M_benchmark(version)