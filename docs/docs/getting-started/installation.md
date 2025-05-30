# Installation Guide

## Requirements

- Python 3.8 or higher
- pip package manager

## Basic Installation

Install Surfing Weights using pip:

```bash
pip install streaming-weights
```

This installs the core package with basic functionality.

## Installation with Extra Features

### Server Components

For running a weight streaming server:

```bash
pip install streaming-weights[server]
```

This includes additional dependencies for:
- FastAPI server
- Monitoring capabilities
- Server-side caching

### Development Installation

For contributing to Surfing Weights or running tests:

```bash
pip install streaming-weights[dev]
```

This includes:
- Testing frameworks
- Development tools
- Documentation dependencies

### Full Installation

To install all optional dependencies:

```bash
pip install streaming-weights[server,dev]
```

## Cloud Storage Support

### Amazon S3

For S3 storage backend support:

```bash
pip install streaming-weights[s3]
```

## Verifying Installation

Test your installation by running:

```python
import streaming_weights
print(streaming_weights.__version__)
```

## Next Steps

- Follow the [Quick Start Guide](quick-start.md) to begin using Surfing Weights
- Learn about [Core Concepts](concepts.md)
- Explore [Configuration Options](../user-guide/configuration.md)