#!/usr/bin/env python3
"""
Setup script for Poker Knight
"""

from setuptools import setup, find_packages
import os

# Read version from VERSION file
def get_version():
    version_file = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_file, 'r') as f:
        return f.read().strip()

# Read long description from README
def get_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(readme_file, 'r', encoding='utf-8') as f:
        return f.read()

setup(
    name="poker-knight",
    version=get_version(),
    author="hildolfr",
    author_email="",
    description="A high-performance Monte Carlo Texas Hold'em poker solver for AI poker players",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/hildolfr/poker_knight",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        # No external dependencies - uses only standard library
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    include_package_data=True,
    package_data={
        "poker_knight": ["config.json"],
        "": ["VERSION", "README.md", "LICENSE"],
    },
    entry_points={
        "console_scripts": [
            "poker-knight=poker_knight:solve_poker_hand",
        ],
    },
    keywords="poker texas-holdem monte-carlo simulation ai game-theory probability",
    project_urls={
        "Bug Reports": "https://github.com/hildolfr/poker_knight/issues",
        "Source": "https://github.com/hildolfr/poker_knight",
        "Documentation": "https://github.com/hildolfr/poker_knight/blob/main/README.md",
        "Changelog": "https://github.com/hildolfr/poker_knight/blob/main/docs/CHANGELOG.md",
    },
) 