# setup.py
# WARNING: This package is in developmental and beta phase. Do NOT use in production environments.
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="streaming-weights",
    version="0.1.0",
    author="Adarsh Reddy",  # <-- Change this to your name
    author_email="adarshadelli@gmail.com",  # <-- Change this to your email
    description="[BETA] Streaming weights engine for edge AI deployment. Do NOT use in production.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/streaming-weights",  # <-- Change this to your repo URL
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "websockets>=10.0",
        "requests",
        "numpy",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio",
            "black",
            "flake8",
            "mypy",
        ],
        "server": [
            "fastapi",
            "uvicorn",
            "aioredis",  # For distributed caching
        ],
    },
    entry_points={
        "console_scripts": [
            "streaming-weights-server=streaming_weights.weight_server:main",
            "streaming-weights-chunk=streaming_weights.chunker:main",
        ],
    },
)
