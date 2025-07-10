#!/bin/bash
# Script to build and upload streaming-weights to PyPI
# Usage: bash publish_to_pypi.sh

set -e

echo "Installing build and twine..."
pip install --upgrade build twine

echo "Cleaning old builds..."
rm -rf dist/ build/ *.egg-info

echo "Building the package..."
python -m build

echo "Checking the distribution..."
twine check dist/*

echo "Uploading to PyPI..."
twine upload dist/*

echo "Done!"
