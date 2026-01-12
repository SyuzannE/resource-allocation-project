"""
Setup configuration for resource-allocation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="resource-allocation-matrix",
    version="1.0.0",
    author="Syuzanna Ghazaryan",
    author_email="syuzanna.ghazaryan@example.com",
    description="Real-time resource allocation using inverse matrices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/resource-allocation-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.4.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
        "benchmark": [
            "matplotlib>=3.7.0",
            "pandas>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resource-allocation=resource_allocation.cli:main",
        ],
    },
)
