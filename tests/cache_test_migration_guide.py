#!/usr/bin/env python3
"""
♞ Poker Knight Cache Test Migration Guide

Guide and utilities for migrating from the old dual-cache test architecture
to the new Phase 4 unified cache system. Provides mapping between old and
new APIs and migration helpers.

Author: hildolfr
License: MIT
"""

import warnings
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Import mappings for migration
OLD_TO_NEW_IMPORTS = {
    # Old dual-cache imports -> New unified cache imports
    'poker_knight.storage.HandCache': 'poker_knight.storage.unified_cache.ThreadSafeMonteCarloCache',
    'poker_knight.storage.PreflopRangeCache': 'poker_knight.storage.preflop_cache.PreflopCache',
    'poker_knight.storage.BoardTextureCache': 'poker_knight.storage.board_cache.BoardScenarioCache',
    'poker_knight.storage.CacheConfig': 'poker_knight.storage.unified_cache.CacheConfig',
    'poker_knight.storage.create_cache_key': 'poker_knight.storage.unified_cache.create_cache_key',
    'poker_knight.storage.cache.HandCache': 'poker_knight.storage.unified_cache.ThreadSafeMonteCarloCache',
    'poker_knight.storage.cache.clear_all_caches': 'poker_knight.storage.unified_cache.clear_unified_cache',
}


@dataclass
class MigrationStatus:
    """Status of test migration."""
    file_path: str
    needs_migration: bool
    old_imports: List[str]
    new_imports: List[str]
    migration_complexity: str  # "simple", "moderate", "complex"
    estimated_effort_hours: float
    notes: List[str]


class CacheTestMigrationHelper:
    """Helper class for migrating cache tests to new architecture."""
    
    def __init__(self):
        self.migration_stats = []
    
    def analyze_test_file(self, file_path: str) -> MigrationStatus:
        """Analyze a test file and determine migration requirements."""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Detect old imports
            old_imports = []
            for old_import in OLD_TO_NEW_IMPORTS.keys():
                if old_import in content:
                    old_imports.append(old_import)
            
            # Determine migration complexity
            complexity = "simple"
            effort = 0.5
            notes = []
            
            if len(old_imports) > 5:
                complexity = "complex"
                effort = 4.0
                notes.append("Many old imports detected - significant refactoring needed")
            elif len(old_imports) > 2:
                complexity = "moderate" 
                effort = 2.0
                notes.append("Moderate refactoring needed")
            
            # Check for specific patterns that increase complexity
            if "HandCache" in content and "PreflopRangeCache" in content:
                complexity = "complex"
                effort = max(effort, 3.0)
                notes.append("Uses dual-cache architecture - needs complete rewrite")
            
            if "cache_stats" in content:
                notes.append("Statistics tracking - verify new stats format")
            
            if "Redis" in content or "SQLite" in content:
                notes.append("Backend-specific tests - map to hierarchical cache layers")
            
            # Generate new imports
            new_imports = [OLD_TO_NEW_IMPORTS.get(old, old) for old in old_imports]
            
            status = MigrationStatus(
                file_path=file_path,
                needs_migration=len(old_imports) > 0,
                old_imports=old_imports,
                new_imports=new_imports,
                migration_complexity=complexity,
                estimated_effort_hours=effort,
                notes=notes
            )
            
            self.migration_stats.append(status)
            return status
            
        except Exception as e:
            return MigrationStatus(
                file_path=file_path,
                needs_migration=False,
                old_imports=[],
                new_imports=[],
                migration_complexity="error",
                estimated_effort_hours=0,
                notes=[f"Error analyzing file: {e}"]
            )
    
    def generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report."""
        total_files = len(self.migration_stats)
        files_needing_migration = len([s for s in self.migration_stats if s.needs_migration])
        
        complexity_breakdown = {}
        for complexity in ["simple", "moderate", "complex", "error"]:
            count = len([s for s in self.migration_stats if s.migration_complexity == complexity])
            complexity_breakdown[complexity] = count
        
        total_effort = sum(s.estimated_effort_hours for s in self.migration_stats)
        
        return {
            'total_files_analyzed': total_files,
            'files_needing_migration': files_needing_migration,
            'migration_percentage': (files_needing_migration / total_files * 100) if total_files > 0 else 0,
            'complexity_breakdown': complexity_breakdown,
            'estimated_total_effort_hours': total_effort,
            'high_priority_files': [
                s.file_path for s in self.migration_stats 
                if s.migration_complexity == "complex"
            ],
            'migration_details': self.migration_stats
        }


# Legacy compatibility wrappers for tests that can't be immediately migrated
class LegacyCacheWrapper:
    """Wrapper to provide legacy cache interface using new cache system."""
    
    def __init__(self):
        warnings.warn(
            "Using legacy cache wrapper. Please migrate to new Phase 4 cache system.",
            DeprecationWarning,
            stacklevel=2
        )
        
        try:
            from poker_knight.storage.unified_cache import ThreadSafeMonteCarloCache
            self._cache = ThreadSafeMonteCarloCache()
        except ImportError:
            self._cache = None
    
    def get_result(self, cache_key):
        """Legacy get_result method."""
        if self._cache:
            return self._cache.get(cache_key)
        return None
    
    def store_result(self, cache_key, result):
        """Legacy store_result method."""
        if self._cache:
            return self._cache.store(cache_key, result)
        return False
    
    def get_stats(self):
        """Legacy get_stats method."""
        if self._cache:
            return self._cache.get_stats()
        return {'total_requests': 0, 'cache_hits': 0, 'cache_misses': 0}
    
    def clear(self):
        """Legacy clear method."""
        if self._cache:
            return self._cache.clear()
        return True


def create_legacy_cache_config(**kwargs):
    """Create legacy cache configuration."""
    warnings.warn(
        "Using legacy cache config. Please migrate to new Phase 4 configuration.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Map legacy config to new config
    try:
        from poker_knight.storage.unified_cache import CacheConfig
        return CacheConfig()
    except ImportError:
        # Return mock config if new system not available
        class MockConfig:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        return MockConfig(**kwargs)


def create_legacy_cache_key(hero_hand, num_opponents, board_cards=None, **kwargs):
    """Create legacy cache key."""
    warnings.warn(
        "Using legacy cache key creation. Please migrate to new CacheKey class.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        from poker_knight.storage.unified_cache import create_cache_key
        return create_cache_key(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board_cards,
            simulation_mode=kwargs.get('simulation_mode', 'default')
        )
    except ImportError:
        # Return simple string key if new system not available
        board_str = "_".join(board_cards) if board_cards else "preflop"
        return f"{hero_hand[0]}{hero_hand[1]}_{num_opponents}_{board_str}"


# Test migration patterns and examples
MIGRATION_PATTERNS = {
    # Import statement migrations
    'import_patterns': {
        'from poker_knight.storage import HandCache': 
            'from poker_knight.storage.unified_cache import ThreadSafeMonteCarloCache as HandCache',
        'from poker_knight.storage import PreflopRangeCache':
            'from poker_knight.storage.preflop_cache import PreflopCache as PreflopRangeCache',
        'from poker_knight.storage import CacheConfig':
            'from poker_knight.storage.unified_cache import CacheConfig',
        'from poker_knight.storage.cache import clear_all_caches':
            'from poker_knight.storage.unified_cache import clear_unified_cache as clear_all_caches',
    },
    
    # Code pattern migrations
    'code_patterns': {
        # Old dual-cache instantiation
        'hand_cache = HandCache(config)': 
            'hand_cache = ThreadSafeMonteCarloCache(max_memory_mb=config.max_memory_mb)',
        
        # Old cache key creation
        'cache_key = create_cache_key(hero_hand, num_opponents, board_cards, config)':
            'cache_key = CacheKey(hero_hand=normalize_hand(hero_hand), num_opponents=num_opponents, board_cards=normalize_board(board_cards), simulation_mode="default")',
        
        # Old cache result storage
        'cache.store_result(key, result_dict)':
            'cache.store(key, CacheResult(**result_dict))',
        
        # Old cache statistics
        'stats = cache.get_stats()':
            'stats = cache.get_stats()  # Returns CacheStats object instead of dict',
    }
}


def print_migration_guide():
    """Print comprehensive migration guide."""
    print("🔄 Cache Test Migration Guide")
    print("=" * 60)
    print()
    
    print("📋 MIGRATION OVERVIEW")
    print("-" * 30)
    print("The cache refactor (Phases 1-4) replaced the old dual-cache architecture")
    print("with a unified hierarchical cache system. This affects:")
    print("• Import statements for cache classes")
    print("• Cache instantiation and configuration") 
    print("• Cache key generation and normalization")
    print("• Cache result storage and retrieval")
    print("• Cache statistics and monitoring")
    print()
    
    print("🚨 BREAKING CHANGES")
    print("-" * 30)
    print("• HandCache → ThreadSafeMonteCarloCache")
    print("• PreflopRangeCache → PreflopCache")
    print("• BoardTextureCache → BoardScenarioCache") 
    print("• Cache keys now use structured CacheKey class")
    print("• Cache results now use structured CacheResult class")
    print("• Statistics format changed from dict to CacheStats object")
    print()
    
    print("📁 MIGRATION PRIORITY")
    print("-" * 30)
    print("HIGH PRIORITY (Complete rewrite needed):")
    print("• test_caching.py - Core cache functionality tests")
    print("• test_storage_cache.py - Comprehensive cache tests")
    print("• test_cache_integration.py - Solver integration tests")
    print()
    print("MEDIUM PRIORITY (Moderate changes needed):")
    print("• test_cache_population.py - Cache population tests")
    print("• test_solver_caching_integration.py - Solver cache tests")
    print("• Redis/SQLite backend tests - Map to hierarchical layers")
    print()
    print("LOW PRIORITY (Minor changes or already updated):")
    print("• test_unified_cache.py - Already tests new system")
    print("• cache_test_base.py - May need updates for new base classes")
    print()
    
    print("🔧 MIGRATION STEPS")
    print("-" * 30)
    print("1. Update import statements using mapping above")
    print("2. Replace cache instantiation with new classes")
    print("3. Update cache key generation to use CacheKey class")
    print("4. Update cache result handling to use CacheResult class")
    print("5. Update test assertions for new statistics format")
    print("6. Add tests for new Phase 4 components if needed")
    print("7. Validate test isolation and cleanup")
    print()
    
    print("💡 MIGRATION TOOLS")
    print("-" * 30)
    print("• Use CacheTestMigrationHelper to analyze files")
    print("• Use LegacyCacheWrapper for temporary compatibility")
    print("• Reference test_phase4_cache_system.py for new patterns")
    print("• Check cache_test_migration_guide.py for mappings")
    print()
    
    print("✅ VALIDATION CHECKLIST")
    print("-" * 30)
    print("□ All import statements updated")
    print("□ Cache instantiation uses new classes")
    print("□ Cache keys use CacheKey class")
    print("□ Cache results use CacheResult class")
    print("□ Test isolation works properly")
    print("□ No references to old dual-cache architecture")
    print("□ Statistics assertions updated for new format")
    print("□ Performance expectations adjusted for hierarchical cache")


if __name__ == "__main__":
    print_migration_guide()
    
    # Example usage of migration helper
    print("\n🔍 EXAMPLE MIGRATION ANALYSIS")
    print("-" * 40)
    
    helper = CacheTestMigrationHelper()
    
    # Analyze a few key test files (if they exist)
    test_files = [
        "tests/test_caching.py",
        "tests/test_storage_cache.py", 
        "tests/test_cache_integration.py",
        "tests/test_unified_cache.py"
    ]
    
    for file_path in test_files:
        try:
            status = helper.analyze_test_file(file_path)
            print(f"\n📄 {file_path}")
            print(f"   Migration needed: {status.needs_migration}")
            print(f"   Complexity: {status.migration_complexity}")
            print(f"   Estimated effort: {status.estimated_effort_hours:.1f} hours")
            if status.notes:
                print(f"   Notes: {'; '.join(status.notes)}")
        except FileNotFoundError:
            print(f"\n📄 {file_path} - File not found")
    
    # Generate overall report
    if helper.migration_stats:
        report = helper.generate_migration_report()
        print(f"\n📊 MIGRATION SUMMARY")
        print(f"Files analyzed: {report['total_files_analyzed']}")
        print(f"Files needing migration: {report['files_needing_migration']}")
        print(f"Total estimated effort: {report['estimated_total_effort_hours']:.1f} hours")
        print(f"Complexity breakdown: {report['complexity_breakdown']}")