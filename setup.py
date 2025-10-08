#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WaveSpectra2Text 安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# 读取requirements文件
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="wavespectra2text",
    version="1.0.0",
    author="WaveSpectra2Text Team",
    author_email="contact@wavespectra2text.com",
    description="双输入语音识别系统 - 支持音频和频谱两种输入模式",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wavespectra2text/wavespectra2text",
    project_urls={
        "Bug Reports": "https://github.com/wavespectra2text/wavespectra2text/issues",
        "Source": "https://github.com/wavespectra2text/wavespectra2text",
        "Documentation": "https://wavespectra2text.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wavespectra2text-train=wavespectra2text.scripts.train:main",
            "wavespectra2text-inference=wavespectra2text.scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "wavespectra2text": ["configs/*.yaml"],
    },
    zip_safe=False,
)
