#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependencies (CMake)
pip install cmake

# Install Python dependencies
pip install -r requirements.txt
