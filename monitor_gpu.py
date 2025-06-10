#!/usr/bin/env python3
"""Monitor GPU utilization during poker solving."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import subprocess
import threading
import time
import numpy as np
from poker_knight import solve_poker_hand

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.utilizations = []
        self.memory_usage = []
        
    def _monitor_loop(self):
        """Monitor GPU stats in a loop."""
        while self.monitoring:
            try:
                # Get GPU stats using nvidia-smi
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    util = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    
                    self.utilizations.append(util)
                    self.memory_usage.append((mem_used, mem_total))
            except:
                pass
            
            time.sleep(0.1)  # Sample every 100ms
    
    def start(self):
        """Start monitoring."""
        self.monitoring = True
        self.utilizations = []
        self.memory_usage = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring and return stats."""
        self.monitoring = False
        self.thread.join()
        
        if self.utilizations:
            return {
                'max_utilization': max(self.utilizations),
                'avg_utilization': np.mean(self.utilizations),
                'max_memory_mb': max(m[0] for m in self.memory_usage),
                'total_memory_mb': self.memory_usage[0][1] if self.memory_usage else 0
            }
        return None

print("=== GPU Utilization Monitor ===\n")

# Test different workloads
tests = [
    ("Small (10K simulations)", 10000, 'fast'),
    ("Medium (100K simulations)", 100000, 'default'),
    ("Large (500K simulations)", 500000, 'precision'),
]

for description, num_sims, mode in tests:
    print(f"\nTest: {description}")
    print("Starting GPU monitoring...")
    
    monitor = GPUMonitor()
    monitor.start()
    
    # Run poker simulation
    start_time = time.time()
    result = solve_poker_hand(['A♠', 'A♥'], 4, simulation_mode=mode)
    elapsed = time.time() - start_time
    
    # Stop monitoring
    stats = monitor.stop()
    
    print(f"Simulation completed in {elapsed*1000:.1f}ms")
    print(f"Win probability: {result.win_probability:.1%}")
    print(f"GPU used: {result.gpu_used}")
    print(f"Backend: {result.backend}")
    
    if stats:
        print(f"\nGPU Stats:")
        print(f"  Max utilization: {stats['max_utilization']}%")
        print(f"  Avg utilization: {stats['avg_utilization']:.1f}%")
        print(f"  Max memory used: {stats['max_memory_mb']} MB / {stats['total_memory_mb']} MB")
    else:
        print("No GPU stats collected")
    
    print("-" * 50)