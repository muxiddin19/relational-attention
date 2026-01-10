"""
Setup script for Relational Attention package.

Install with:
    pip install -e .

Or for development:
    pip install -e ".[dev]"
"""

from setuptools import setup, find_packages
import os

# Read README from the package directory
readme_path = os.path.join(os.path.dirname(__file__), "relational_attention", "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Relational Attention: A Set-Theoretic Foundation for Neural Structured Reasoning"

setup(
    name="relational-attention",
    version="0.1.0",
    author="Anonymous",
    author_email="anonymous@example.com",
    description="Relational Attention: A Set-Theoretic Foundation for Neural Structured Reasoning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/muxiddin19/relational-attention",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*", "emnlp26", "emnlp26.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
        ],
    },
    include_package_data=True,
    keywords=[
        "attention",
        "transformer",
        "relational-algebra",
        "structured-reasoning",
        "text-to-sql",
        "deep-learning",
        "pytorch",
    ],
    project_urls={
        "Paper": "https://anonymous.4open.science/r/relational-attention/",
        "Source": "https://github.com/muxiddin19/relational-attention",
    },
)
