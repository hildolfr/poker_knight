"""
Global pytest configuration and fixtures.
Platform-agnostic test configuration.
"""

import sys
import pytest
import os

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def pytest_configure(config):
    """Configure pytest with platform-agnostic settings."""
    # Set a reasonable timeout for all tests
    # This prevents hanging tests on any platform
    if not hasattr(config.option, 'timeout'):
        config.option.timeout = 300  # 5 minutes max per test
    
    # Ensure we use the correct Python interpreter
    os.environ['PYTEST_CURRENT_TEST_PYTHON'] = sys.executable

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add platform-specific markers."""
    for item in items:
        # Mark tests that might be slow on certain platforms
        if 'edge_cases_extended' in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        
        # Skip certain tests if running in CI or resource-constrained environments
        if os.environ.get('CI') or os.environ.get('PYTEST_FAST_MODE'):
            if 'slow' in item.keywords or 'performance' in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Skipping slow tests in CI/fast mode"))