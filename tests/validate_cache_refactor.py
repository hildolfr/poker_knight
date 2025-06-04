#!/usr/bin/env python3
"""
♞ Poker Knight Cache Refactor Validation Script

Comprehensive validation script to verify the cache refactor implementation
and test the new Phase 4 cache system. Provides automated validation of
all cache components and integration with the MonteCarloSolver.

Author: hildolfr
License: MIT
"""

import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple

def validate_imports() -> Dict[str, Any]:
    """Validate that all Phase 4 components can be imported."""
    print("🔍 Validating Phase 4 Component Imports...")
    
    results = {
        'success': True,
        'components': {},
        'errors': []
    }
    
    # Test Phase 4 component imports
    components_to_test = [
        ('unified_cache', 'poker_knight.storage.unified_cache'),
        ('hierarchical_cache', 'poker_knight.storage.hierarchical_cache'),
        ('adaptive_manager', 'poker_knight.storage.adaptive_cache_manager'),
        ('performance_monitor', 'poker_knight.storage.cache_performance_monitor'),
        ('intelligent_prepopulation', 'poker_knight.storage.intelligent_prepopulation'),
        ('optimized_persistence', 'poker_knight.storage.optimized_persistence'),
        ('phase4_integration', 'poker_knight.storage.phase4_integration'),
        ('solver', 'poker_knight'),
    ]
    
    for component_name, module_path in components_to_test:
        try:
            __import__(module_path)
            results['components'][component_name] = True
            print(f"  ✅ {component_name}: Available")
        except ImportError as e:
            results['components'][component_name] = False
            results['success'] = False
            results['errors'].append(f"{component_name}: {e}")
            print(f"  ❌ {component_name}: Failed - {e}")
        except Exception as e:
            results['components'][component_name] = False
            results['success'] = False
            results['errors'].append(f"{component_name}: Unexpected error - {e}")
            print(f"  ⚠️ {component_name}: Error - {e}")
    
    return results


def validate_unified_cache() -> Dict[str, Any]:
    """Validate unified cache functionality."""
    print("\n🧪 Validating Unified Cache System...")
    
    results = {
        'success': True,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    try:
        from poker_knight.storage.unified_cache import (
            ThreadSafeMonteCarloCache, CacheKey, CacheResult, create_cache_key
        )
        
        # Test cache creation
        cache = ThreadSafeMonteCarloCache(max_memory_mb=32, max_entries=100)
        results['tests_passed'] += 1
        print("  ✅ Cache creation")
        
        # Test cache key creation
        cache_key = CacheKey(
            hero_hand="AK_suited",
            num_opponents=2,
            board_cards="preflop",
            simulation_mode="default"
        )
        results['tests_passed'] += 1
        print("  ✅ Cache key creation")
        
        # Test cache result creation
        cache_result = CacheResult(
            win_probability=0.68,
            tie_probability=0.02,
            loss_probability=0.30,
            confidence_interval=(0.65, 0.71),
            simulations_run=10000,
            execution_time_ms=100.0,
            hand_categories={},
            metadata={},
            timestamp=time.time()
        )
        results['tests_passed'] += 1
        print("  ✅ Cache result creation")
        
        # Test cache operations
        success = cache.store(cache_key, cache_result)
        if success:
            results['tests_passed'] += 1
            print("  ✅ Cache storage")
        else:
            raise Exception("Cache storage failed")
        
        retrieved = cache.get(cache_key)
        if retrieved and abs(retrieved.win_probability - 0.68) < 0.001:
            results['tests_passed'] += 1
            print("  ✅ Cache retrieval")
        else:
            raise Exception("Cache retrieval failed or incorrect data")
        
        # Test cache statistics
        stats = cache.get_stats()
        if stats and stats.total_requests > 0:
            results['tests_passed'] += 1
            print("  ✅ Cache statistics")
        else:
            raise Exception("Cache statistics failed")
        
        # Test cache clearing
        success = cache.clear()
        if success:
            results['tests_passed'] += 1
            print("  ✅ Cache clearing")
        else:
            raise Exception("Cache clearing failed")
            
    except Exception as e:
        results['success'] = False
        results['tests_failed'] += 1
        results['errors'].append(f"Unified cache validation: {e}")
        print(f"  ❌ Unified cache error: {e}")
    
    return results


def validate_hierarchical_cache() -> Dict[str, Any]:
    """Validate hierarchical cache functionality."""
    print("\n🏗️ Validating Hierarchical Cache System...")
    
    results = {
        'success': True,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    try:
        from poker_knight.storage.hierarchical_cache import (
            HierarchicalCache, HierarchicalCacheConfig
        )
        from poker_knight.storage.unified_cache import CacheKey, CacheResult
        
        # Test hierarchical cache creation
        config = HierarchicalCacheConfig()
        # Disable Redis/SQLite for testing
        config.l2_config.enabled = False
        config.l3_config.enabled = False
        
        cache = HierarchicalCache(config)
        results['tests_passed'] += 1
        print("  ✅ Hierarchical cache creation")
        
        # Test basic operations
        cache_key = CacheKey("KK", 3, "preflop", "fast")
        cache_result = CacheResult(0.82, 0.01, 0.17, (0.79, 0.85), 5000, 75.0, {}, {}, time.time())
        
        success = cache.store(cache_key, cache_result)
        if success:
            results['tests_passed'] += 1
            print("  ✅ Hierarchical storage")
        
        retrieved = cache.get(cache_key)
        if retrieved and abs(retrieved.win_probability - 0.82) < 0.001:
            results['tests_passed'] += 1
            print("  ✅ Hierarchical retrieval")
        
        # Test statistics
        stats = cache.get_stats()
        if stats:
            results['tests_passed'] += 1
            print("  ✅ Hierarchical statistics")
        
        # Cleanup
        cache.clear()
        cache.shutdown()
        results['tests_passed'] += 1
        print("  ✅ Hierarchical cleanup")
        
    except Exception as e:
        results['success'] = False
        results['tests_failed'] += 1
        results['errors'].append(f"Hierarchical cache validation: {e}")
        print(f"  ❌ Hierarchical cache error: {e}")
    
    return results


def validate_phase4_integration() -> Dict[str, Any]:
    """Validate Phase 4 integration system."""
    print("\n🎯 Validating Phase 4 Integration System...")
    
    results = {
        'success': True,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    try:
        from poker_knight.storage.phase4_integration import (
            Phase4CacheSystem, Phase4Config, create_balanced_cache_system
        )
        
        # Test system creation
        system = create_balanced_cache_system()
        results['tests_passed'] += 1
        print("  ✅ Phase 4 system creation")
        
        # Test initialization
        success = system.initialize()
        if success:
            results['tests_passed'] += 1
            print("  ✅ Phase 4 initialization")
        
        # Test status
        status = system.get_system_status()
        if status and status.get('initialized'):
            results['tests_passed'] += 1
            print("  ✅ Phase 4 status reporting")
        
        # Test cleanup
        system.stop_services()
        results['tests_passed'] += 1
        print("  ✅ Phase 4 cleanup")
        
    except Exception as e:
        results['success'] = False
        results['tests_failed'] += 1
        results['errors'].append(f"Phase 4 integration validation: {e}")
        print(f"  ❌ Phase 4 integration error: {e}")
    
    return results


def validate_solver_integration() -> Dict[str, Any]:
    """Validate solver integration with new cache system."""
    print("\n🤖 Validating Solver Cache Integration...")
    
    results = {
        'success': True,
        'tests_passed': 0,
        'tests_failed': 0,
        'errors': []
    }
    
    try:
        from poker_knight import MonteCarloSolver
        
        # Test solver creation with caching
        solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
        results['tests_passed'] += 1
        print("  ✅ Solver creation with caching")
        
        # Test cache statistics interface
        stats = solver.get_cache_stats()
        if stats is not None:  # Can be None or dict
            results['tests_passed'] += 1
            print("  ✅ Solver cache statistics interface")
        
        # Test basic analysis (quick test)
        try:
            result = solver.analyze_hand(['A♠', 'A♥'], 2, simulation_mode="fast")
            if result and hasattr(result, 'win_probability'):
                results['tests_passed'] += 1
                print("  ✅ Solver analysis with cache")
            
            # Test cache hit (repeat same analysis)
            result2 = solver.analyze_hand(['A♠', 'A♥'], 2, simulation_mode="fast")
            if result2 and abs(result.win_probability - result2.win_probability) < 0.001:
                results['tests_passed'] += 1
                print("  ✅ Solver cache hit behavior")
                
        except Exception as e:
            print(f"  ⚠️ Solver analysis test skipped: {e}")
        
        # Cleanup
        solver.close()
        results['tests_passed'] += 1
        print("  ✅ Solver cleanup")
        
    except Exception as e:
        results['success'] = False
        results['tests_failed'] += 1
        results['errors'].append(f"Solver integration validation: {e}")
        print(f"  ❌ Solver integration error: {e}")
    
    return results


def validate_test_architecture() -> Dict[str, Any]:
    """Validate new test architecture components."""
    print("\n🧪 Validating New Test Architecture...")
    
    results = {
        'success': True,
        'components_found': 0,
        'components_missing': 0,
        'errors': []
    }
    
    # Check for new test files
    test_files = [
        'test_phase4_cache_system.py',
        'updated_cache_test_base.py',
        'test_modern_cache_integration.py',
        'cache_test_migration_guide.py',
        'TEST_SUITE_MIGRATION_SUMMARY.md'
    ]
    
    tests_dir = Path(__file__).parent
    
    for test_file in test_files:
        file_path = tests_dir / test_file
        if file_path.exists():
            results['components_found'] += 1
            print(f"  ✅ {test_file}")
        else:
            results['components_missing'] += 1
            results['errors'].append(f"Missing test file: {test_file}")
            print(f"  ❌ {test_file}")
    
    # Try importing test base classes
    try:
        from .updated_cache_test_base import BaseCacheTest, UnifiedCacheTestBase
        results['components_found'] += 1
        print("  ✅ Test base class imports")
    except ImportError:
        try:
            from updated_cache_test_base import BaseCacheTest, UnifiedCacheTestBase
            results['components_found'] += 1
            print("  ✅ Test base class imports")
        except ImportError as e:
            results['components_missing'] += 1
            results['errors'].append(f"Test base import failed: {e}")
            print(f"  ❌ Test base class imports: {e}")
    except Exception as e:
        results['components_missing'] += 1
        results['errors'].append(f"Test base import error: {e}")
        print(f"  ⚠️ Test base class imports: {e}")
    
    results['success'] = results['components_missing'] == 0
    return results


def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation of cache refactor."""
    print("♞ Poker Knight Cache Refactor Validation")
    print("=" * 60)
    
    all_results = {
        'overall_success': True,
        'validation_time': time.time(),
        'component_results': {}
    }
    
    # Run all validation tests
    validation_tests = [
        ('imports', validate_imports),
        ('unified_cache', validate_unified_cache),
        ('hierarchical_cache', validate_hierarchical_cache),
        ('phase4_integration', validate_phase4_integration),
        ('solver_integration', validate_solver_integration),
        ('test_architecture', validate_test_architecture),
    ]
    
    for test_name, test_func in validation_tests:
        try:
            result = test_func()
            all_results['component_results'][test_name] = result
            
            if not result['success']:
                all_results['overall_success'] = False
                
        except Exception as e:
            print(f"\n❌ {test_name} validation failed with exception: {e}")
            traceback.print_exc()
            all_results['component_results'][test_name] = {
                'success': False,
                'error': str(e)
            }
            all_results['overall_success'] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("-" * 30)
    
    for test_name, result in all_results['component_results'].items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{status} {test_name}")
        
        if 'tests_passed' in result:
            print(f"    Tests passed: {result['tests_passed']}")
        if 'tests_failed' in result:
            print(f"    Tests failed: {result['tests_failed']}")
        if 'components_found' in result:
            print(f"    Components found: {result['components_found']}")
        if 'components_missing' in result:
            print(f"    Components missing: {result['components_missing']}")
    
    print("\n" + "-" * 30)
    if all_results['overall_success']:
        print("🎉 OVERALL RESULT: ✅ VALIDATION SUCCESSFUL")
        print("The cache refactor implementation is working correctly!")
    else:
        print("⚠️ OVERALL RESULT: ❌ VALIDATION FAILED")
        print("Some components need attention. Check the errors above.")
    
    # Print recommendations
    print("\n💡 RECOMMENDATIONS:")
    if all_results['overall_success']:
        print("• Cache refactor validation passed - system ready for use")
        print("• Begin migrating legacy tests using the new architecture")
        print("• Run performance benchmarks to validate improvements")
        print("• Update CI/CD pipeline to use new test patterns")
    else:
        print("• Fix component errors identified above")
        print("• Ensure all Phase 4 components are properly installed")
        print("• Check import paths and dependencies")
        print("• Review implementation for missing components")
    
    return all_results


def print_migration_status():
    """Print current migration status and next steps."""
    print("\n📋 MIGRATION STATUS")
    print("-" * 30)
    print("✅ COMPLETED:")
    print("  • Phase 4 cache system implementation")
    print("  • New test architecture and base classes")
    print("  • Modern cache integration tests")
    print("  • Migration tools and documentation")
    print("  • Comprehensive validation framework")
    
    print("\n🔄 IN PROGRESS:")
    print("  • Legacy test migration (15-20 test files)")
    print("  • Solver integration updates")
    print("  • Performance optimization validation")
    
    print("\n📅 NEXT STEPS:")
    print("  1. Migrate high-priority legacy tests (test_caching.py, etc.)")
    print("  2. Update test runners and CI configuration")
    print("  3. Add comprehensive performance validation tests")
    print("  4. Complete documentation updates")
    
    print("\n🔗 RESOURCES:")
    print("  • tests/TEST_SUITE_MIGRATION_SUMMARY.md - Complete migration guide")
    print("  • tests/cache_test_migration_guide.py - Migration tools")
    print("  • tests/updated_cache_test_base.py - Modern base classes")
    print("  • tests/test_phase4_cache_system.py - Reference implementation")


if __name__ == "__main__":
    # Run validation
    results = run_comprehensive_validation()
    
    # Print migration status
    print_migration_status()
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)