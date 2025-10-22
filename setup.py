"""
Setup script for Trajectory-Guided LTX Video Training
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ltxv-trajectory-training",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Complete system for training LTX Video with trajectory guidance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/ltxv-trajectory-training",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
        ],
        "quality": [
            "scikit-image>=0.21.0",
            "lpips>=0.1.4",
        ]
    },
    entry_points={
        "console_scripts": [
            "ltxv-prepare-dataset=scripts.prepare_dataset:main",
            "ltxv-train=src.training.train_trajectory_iclora:main",
            "ltxv-inference=src.inference.trajectory_guided_inference:main",
        ],
    },
)
