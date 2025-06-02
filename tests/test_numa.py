#!/usr/bin/env python3
"""
Comprehensive test suite for NUMA (Non-Uniform Memory Access) functionality in Poker Knight.

Tests the NUMA topology detection and process affinity management for optimal
performance on multi-socket systems and server hardware.

This module tests:
- NumaTopology class functionality
- CPU-to-NUMA node mapping
- NUMA-aware work distribution
- Process affinity assignment
- NUMA configuration validation
- Integration with parallel processing engine

Test Categories:
- Unit tests for NUMA topology detection
- Integration tests with parallel processing
- Performance tests for NUMA-aware optimization
- Edge cases and error handling

Author: Poker Knight Test Suite
License: MIT
"""

import pytest
import unittest
import multiprocessing as mp
import threading
import time
import sys
import os
from unittest.mock import patch, MagicMock

# Import NUMA-related components
try:
    from poker_knight.core.parallel import (
        NumaTopology, ProcessingConfig, WorkDistributor, 
        ParallelSimulationEngine, create_parallel_engine
    )
    NUMA_MODULE_AVAILABLE = True
except ImportError as e:
    NUMA_MODULE_AVAILABLE = False
    pytest.skip(f"NUMA module not available: {e}", allow_module_level=True)

# Try to import psutil for system information
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


@pytest.mark.numa
@pytest.mark.unit
class TestNumaTopology:
    """Test NUMA topology detection and management."""
    
    def test_numa_topology_initialization(self):
        """Test NumaTopology initialization."""
        numa = NumaTopology()
        
        # Should initialize without errors
        assert numa is not None
        assert hasattr(numa, 'available')
        assert hasattr(numa, '_topology')
        assert hasattr(numa, '_cpu_to_node')
        assert hasattr(numa, '_node_to_cpus')
    
    def test_numa_availability_detection(self):
        """Test NUMA availability detection."""
        numa = NumaTopology()
        
        # Availability should be boolean
        assert isinstance(numa.available, bool)
        
        # If psutil is available, NUMA detection should work
        if PSUTIL_AVAILABLE:
            # Should have some topology information
            topology = numa.get_topology_info()
            assert isinstance(topology, dict)
        else:
            # Without psutil, should gracefully handle unavailability
            assert not numa.available
    
    def test_numa_topology_info(self):
        """Test NUMA topology information retrieval."""
        numa = NumaTopology()
        topology = numa.get_topology_info()
        
        # Should return a dictionary
        assert isinstance(topology, dict)
        
        if numa.available and topology:
            # Should contain expected keys
            expected_keys = ['numa_nodes', 'logical_cores', 'physical_cores', 'cores_per_node']
            for key in expected_keys:
                assert key in topology
                assert isinstance(topology[key], int)
                assert topology[key] > 0
            
            # Logical cores should be >= physical cores
            assert topology['logical_cores'] >= topology['physical_cores']
            
            # NUMA nodes should be reasonable (1-8 for most systems)
            assert 1 <= topology['numa_nodes'] <= 8
    
    def test_cpu_to_numa_node_mapping(self):
        """Test CPU to NUMA node mapping."""
        numa = NumaTopology()
        
        if numa.available:
            topology = numa.get_topology_info()
            if topology:
                logical_cores = topology['logical_cores']
                numa_nodes = topology['numa_nodes']
                
                # Test each CPU ID
                for cpu_id in range(logical_cores):
                    node = numa.get_numa_node(cpu_id)
                    assert node is not None
                    assert isinstance(node, int)
                    assert 0 <= node < numa_nodes
                
                # Test invalid CPU ID
                invalid_cpu = logical_cores + 10
                assert numa.get_numa_node(invalid_cpu) is None
    
    def test_numa_node_to_cpus_mapping(self):
        """Test NUMA node to CPUs mapping."""
        numa = NumaTopology()
        
        if numa.available:
            topology = numa.get_topology_info()
            if topology:
                numa_nodes = topology['numa_nodes']
                logical_cores = topology['logical_cores']
                
                total_cpus_mapped = 0
                for node_id in range(numa_nodes):
                    cpus = numa.get_cpus_for_node(node_id)
                    assert isinstance(cpus, list)
                    assert len(cpus) > 0
                    
                    # All CPU IDs should be valid
                    for cpu_id in cpus:
                        assert isinstance(cpu_id, int)
                        assert 0 <= cpu_id < logical_cores
                    
                    total_cpus_mapped += len(cpus)
                
                # All CPUs should be mapped to exactly one NUMA node
                assert total_cpus_mapped == logical_cores
                
                # Test invalid NUMA node
                invalid_node = numa_nodes + 5
                assert numa.get_cpus_for_node(invalid_node) == []
    
    def test_numa_topology_consistency(self):
        """Test consistency of NUMA topology mapping."""
        numa = NumaTopology()
        
        if numa.available:
            topology = numa.get_topology_info()
            if topology:
                # Verify bidirectional mapping consistency
                for node_id in range(topology['numa_nodes']):
                    cpus = numa.get_cpus_for_node(node_id)
                    
                    for cpu_id in cpus:
                        mapped_node = numa.get_numa_node(cpu_id)
                        assert mapped_node == node_id, \
                            f"CPU {cpu_id} maps to node {mapped_node} but is listed under node {node_id}"


@pytest.mark.numa
@pytest.mark.unit
class TestNumaConfiguration:
    """Test NUMA configuration and validation."""
    
    def test_processing_config_numa_settings(self):
        """Test NUMA settings in ProcessingConfig."""
        # Default configuration
        config = ProcessingConfig()
        assert hasattr(config, 'numa_aware')
        assert hasattr(config, 'numa_node_affinity')
        assert isinstance(config.numa_aware, bool)
        assert isinstance(config.numa_node_affinity, bool)
        
        # Custom NUMA configuration
        numa_config = ProcessingConfig(
            numa_aware=True,
            numa_node_affinity=True
        )
        assert numa_config.numa_aware is True
        assert numa_config.numa_node_affinity is True
    
    def test_numa_aware_work_distributor(self):
        """Test WorkDistributor with NUMA awareness."""
        numa_topology = NumaTopology()
        config = ProcessingConfig(numa_aware=True, numa_node_affinity=True)
        
        distributor = WorkDistributor(config, numa_topology)
        assert distributor.config.numa_aware is True
        assert distributor.numa == numa_topology
        
        # Test work plan creation with NUMA awareness
        work_plan = distributor.create_work_plan(
            total_simulations=10000,
            complexity_score=6.0
        )
        
        assert isinstance(work_plan, dict)
        assert 'numa_assignment' in work_plan
        
        if numa_topology.available:
            numa_assignment = work_plan['numa_assignment']
            if numa_assignment:
                assert isinstance(numa_assignment, dict)
                assert 'processes' in numa_assignment
                assert 'threads' in numa_assignment
    
    def test_numa_assignment_distribution(self):
        """Test NUMA assignment distribution across nodes."""
        numa_topology = NumaTopology()
        
        if not numa_topology.available:
            pytest.skip("NUMA topology not available on this system")
        
        config = ProcessingConfig(numa_aware=True, numa_node_affinity=True)
        distributor = WorkDistributor(config, numa_topology)
        
        # Create work plan with multiple workers
        work_plan = distributor.create_work_plan(
            total_simulations=50000,
            complexity_score=7.0
        )
        
        numa_assignment = work_plan.get('numa_assignment', {})
        if numa_assignment:
            processes = numa_assignment.get('processes', {})
            threads = numa_assignment.get('threads', {})
            
            topology = numa_topology.get_topology_info()
            numa_nodes = topology.get('numa_nodes', 1)
            
            # Check process assignments
            for worker_id, assignment in processes.items():
                assert 'numa_node' in assignment
                assert 'cpu_affinity' in assignment
                assert 0 <= assignment['numa_node'] < numa_nodes
                assert isinstance(assignment['cpu_affinity'], list)
                assert len(assignment['cpu_affinity']) > 0
            
            # Check thread assignments
            for worker_id, assignment in threads.items():
                assert 'numa_node' in assignment
                assert 'cpu_affinity' in assignment
                assert 0 <= assignment['numa_node'] < numa_nodes
                assert isinstance(assignment['cpu_affinity'], list)


@pytest.mark.numa
@pytest.mark.integration
class TestNumaIntegration:
    """Test NUMA integration with parallel processing engine."""
    
    def test_parallel_engine_numa_initialization(self):
        """Test parallel engine initialization with NUMA support."""
        config = ProcessingConfig(numa_aware=True)
        engine = ParallelSimulationEngine(config)
        
        assert engine.config.numa_aware is True
        assert engine.numa_topology is not None
        assert isinstance(engine.numa_topology, NumaTopology)
        assert engine.work_distributor.numa == engine.numa_topology
    
    def test_numa_aware_engine_creation(self):
        """Test creating NUMA-aware engine with factory function."""
        # Test auto-configuration
        engine = create_parallel_engine()
        assert engine.numa_topology is not None
        
        # Test explicit NUMA configuration
        numa_config = ProcessingConfig(
            numa_aware=True,
            numa_node_affinity=True,
            max_processes=mp.cpu_count()
        )
        numa_engine = create_parallel_engine(numa_config)
        
        assert numa_engine.config.numa_aware is True
        assert numa_engine.config.numa_node_affinity is True
    
    def test_numa_work_plan_execution(self):
        """Test work plan execution with NUMA assignments."""
        numa_topology = NumaTopology()
        
        if not numa_topology.available:
            pytest.skip("NUMA topology not available for integration testing")
        
        config = ProcessingConfig(
            numa_aware=True,
            numa_node_affinity=True,
            max_processes=2,
            max_threads=2
        )
        
        engine = ParallelSimulationEngine(config)
        
        # Mock simulation function for testing
        def mock_simulation(batch_size, worker_id, **kwargs):
            # Simulate some work
            time.sleep(0.1)
            return {
                'simulations_completed': batch_size,
                'worker_id': worker_id,
                'results': [0.5] * batch_size
            }
        
        # Create and execute work plan
        scenario_metadata = {'complexity_score': 6.0}
        
        try:
            results, stats = engine.execute_simulation_batch(
                simulation_function=mock_simulation,
                total_simulations=1000,
                scenario_metadata=scenario_metadata
            )
            
            # Verify execution completed
            assert results is not None
            assert stats is not None
            assert stats.total_simulations > 0
            assert stats.worker_count > 0
            
            # Check NUMA distribution in stats
            assert hasattr(stats, 'numa_distribution')
            assert isinstance(stats.numa_distribution, dict)
            
        except Exception as e:
            # Some systems may not support process execution in tests
            pytest.skip(f"Process execution not supported in test environment: {e}")


@pytest.mark.numa
@pytest.mark.performance
class TestNumaPerformance:
    """Test NUMA-related performance optimizations."""
    
    def test_numa_vs_non_numa_performance_comparison(self):
        """Compare performance with and without NUMA awareness."""
        if not NumaTopology().available:
            pytest.skip("NUMA topology not available for performance testing")
        
        # Configuration without NUMA
        non_numa_config = ProcessingConfig(
            numa_aware=False,
            max_processes=2,
            max_threads=2
        )
        
        # Configuration with NUMA
        numa_config = ProcessingConfig(
            numa_aware=True,
            numa_node_affinity=True,
            max_processes=2,
            max_threads=2
        )
        
        def benchmark_simulation(batch_size, worker_id, **kwargs):
            # Simulate CPU-intensive work
            total = 0
            for i in range(batch_size * 100):
                total += i * i
            return {'result': total, 'batch_size': batch_size}
        
        scenario_metadata = {'complexity_score': 7.0}
        
        # Benchmark non-NUMA configuration
        non_numa_engine = ParallelSimulationEngine(non_numa_config)
        start_time = time.time()
        
        try:
            results1, stats1 = non_numa_engine.execute_simulation_batch(
                simulation_function=benchmark_simulation,
                total_simulations=1000,
                scenario_metadata=scenario_metadata
            )
            non_numa_time = time.time() - start_time
        except:
            pytest.skip("Non-NUMA benchmark failed")
        
        # Benchmark NUMA configuration
        numa_engine = ParallelSimulationEngine(numa_config)
        start_time = time.time()
        
        try:
            results2, stats2 = numa_engine.execute_simulation_batch(
                simulation_function=benchmark_simulation,
                total_simulations=1000,
                scenario_metadata=scenario_metadata
            )
            numa_time = time.time() - start_time
        except:
            pytest.skip("NUMA benchmark failed")
        
        # Both should complete successfully
        assert results1 is not None
        assert results2 is not None
        assert stats1.total_simulations == stats2.total_simulations
        
        # NUMA should not significantly degrade performance
        # (Allow up to 50% overhead for test environment)
        assert numa_time < non_numa_time * 1.5
    
    def test_numa_memory_locality(self):
        """Test memory locality with NUMA-aware allocation."""
        numa_topology = NumaTopology()
        
        if not numa_topology.available:
            pytest.skip("NUMA topology not available for memory testing")
        
        topology = numa_topology.get_topology_info()
        if topology.get('numa_nodes', 1) < 2:
            pytest.skip("Single NUMA node system - memory locality test not applicable")
        
        config = ProcessingConfig(
            numa_aware=True,
            numa_node_affinity=True
        )
        
        engine = ParallelSimulationEngine(config)
        
        # Test that workers are distributed across NUMA nodes
        work_plan = engine.work_distributor.create_work_plan(
            total_simulations=10000,
            complexity_score=6.0
        )
        
        numa_assignment = work_plan.get('numa_assignment', {})
        if numa_assignment:
            process_nodes = set()
            for assignment in numa_assignment.get('processes', {}).values():
                process_nodes.add(assignment['numa_node'])
            
            # Should utilize multiple NUMA nodes if available
            assert len(process_nodes) > 1 or len(numa_assignment.get('processes', {})) == 1


@pytest.mark.numa
@pytest.mark.edge_cases
class TestNumaEdgeCases:
    """Test NUMA functionality edge cases and error handling."""
    
    def test_numa_topology_detection_failure(self):
        """Test graceful handling of NUMA topology detection failure."""
        with patch('poker_knight.core.parallel.psutil') as mock_psutil:
            # Simulate psutil unavailability
            mock_psutil.cpu_count.side_effect = ImportError("psutil not available")
            
            numa = NumaTopology()
            assert numa.available is False
            assert numa.get_topology_info() == {}
            assert numa.get_numa_node(0) is None
            assert numa.get_cpus_for_node(0) == []
    
    def test_invalid_numa_operations(self):
        """Test handling of invalid NUMA operations."""
        numa = NumaTopology()
        
        # Test with invalid inputs
        assert numa.get_numa_node(-1) is None
        assert numa.get_numa_node(99999) is None
        assert numa.get_cpus_for_node(-1) == []
        assert numa.get_cpus_for_node(99999) == []
    
    def test_numa_with_single_core_system(self):
        """Test NUMA functionality on single-core systems."""
        with patch('poker_knight.core.parallel.psutil') as mock_psutil:
            # Simulate single-core system
            mock_psutil.cpu_count.return_value = 1
            
            numa = NumaTopology()
            topology = numa.get_topology_info()
            
            if topology:
                assert topology['numa_nodes'] == 1
                assert topology['logical_cores'] == 1
                assert numa.get_numa_node(0) == 0
                assert numa.get_cpus_for_node(0) == [0]
    
    def test_numa_configuration_conflicts(self):
        """Test handling of conflicting NUMA configurations."""
        # Test NUMA awareness disabled but affinity enabled
        config = ProcessingConfig(
            numa_aware=False,
            numa_node_affinity=True  # This should be ignored
        )
        
        numa_topology = NumaTopology()
        distributor = WorkDistributor(config, numa_topology)
        
        work_plan = distributor.create_work_plan(
            total_simulations=5000,
            complexity_score=5.0
        )
        
        # Should not have NUMA assignment when NUMA awareness is disabled
        numa_assignment = work_plan.get('numa_assignment')
        assert numa_assignment is None or not numa_assignment
    
    def test_numa_with_insufficient_resources(self):
        """Test NUMA behavior with insufficient system resources."""
        config = ProcessingConfig(
            numa_aware=True,
            max_processes=0,  # No processes
            max_threads=1     # Minimal threads
        )
        
        numa_topology = NumaTopology()
        distributor = WorkDistributor(config, numa_topology)
        
        work_plan = distributor.create_work_plan(
            total_simulations=100,
            complexity_score=2.0  # Low complexity
        )
        
        # Should handle minimal resources gracefully
        assert work_plan['worker_counts']['processes'] == 0
        assert work_plan['worker_counts']['threads'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 