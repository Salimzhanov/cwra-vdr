#!/usr/bin/env python3
"""
Setup script for CWRA (Calibrated Weighted Rank Aggregation)
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cwra-vdr",
    version="1.0.0",
    author="Abylay Salimzhanov, Ferdinand Molnár, Siamac Fazli",
    author_email="",
    description="Calibrated Weighted Rank Aggregation for VDR Virtual Screening",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cwra-vdr-benchmark",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
    entry_points={
        "console_scripts": [
            "cwra=cwra:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)