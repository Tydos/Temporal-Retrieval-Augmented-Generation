#!/bin/bash

# Step 1: Environment Setup and Dependency Installation

echo "Setting up Python environment and installing dependencies..."

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Environment setup - Project root: $PROJECT_ROOT"
cd "$PROJECT_ROOT"
echo "Changed to project directory: $(pwd)"

# Check Python version
python_version=$(python3 --version 2>&1)
echo "Using Python: $python_version"

# Skip virtual environment creation for now
echo "Skipping virtual environment creation (will be created later if needed)"
echo "Installing packages globally for now..."

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install additional development dependencies
echo "Installing additional development tools..."
pip install jupyter ipykernel

# Set up pre-commit hooks (optional)
echo "Setting up development tools..."
pip install pre-commit black flake8

# Download NLTK data (required for some evaluations)
echo "Downloading NLTK data..."
python -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('NLTK data downloaded successfully')
except Exception as e:
    print(f'Note: NLTK download failed: {e}')
"

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('CUDA not available, will use CPU')
"

# Set up Weights & Biases (optional)
echo "Setting up Weights & Biases (optional)..."
if command -v wandb &> /dev/null; then
    echo "wandb is already installed"
else
    pip install wandb
fi

echo "Note: To use wandb logging, run 'wandb login' and set your API key"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{chroniclingqa,temprageval,generated_questions,atlas_2021,fineweb}
mkdir -p models/{cache,time_aware_contriever}
mkdir -p outputs/{chroniclingqa,temprageval,mrag}
mkdir -p logs

# Download FineWeb dataset (matches notebook exactly)
echo ""
echo "Downloading and processing FineWeb dataset..."
echo "This will collect 500,000 temporal passages from FineWeb-edu (this may take some time)..."
python src/fineweb_loader.py

echo "Environment setup completed successfully!"
echo "Using Python at: $(which python3)"
echo ""
echo "Note: Packages installed globally. You can create a virtual environment later if needed."
echo ""
echo "Next step: Run ./scripts/02_generate_questions.sh"