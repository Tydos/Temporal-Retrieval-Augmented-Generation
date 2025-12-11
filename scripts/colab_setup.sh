#!/bin/bash

# Colab-Specific Setup for Time-Aware RAG
# This script is optimized for Google Colab environment

echo "=========================================="
echo "TIME-AWARE RAG SETUP FOR GOOGLE COLAB"
echo "=========================================="

# Set working directory
cd /content

# Clone or ensure we have the project
if [ ! -d "TimeAwareRAG_Final" ]; then
    echo "Project directory not found. Please ensure the files are uploaded to /content/TimeAwareRAG_Final"
    exit 1
fi

cd TimeAwareRAG_Final

# Install dependencies directly with pip3
echo "Installing required packages for Colab..."
pip3 install --upgrade pip

# Install core ML packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install transformers and related packages
pip3 install transformers==4.36.0
pip3 install datasets==2.14.6
pip3 install accelerate==0.24.1
pip3 install sentence-transformers==2.2.2

# Install data processing packages
pip3 install pandas numpy tqdm jsonlines

# Install evaluation packages
pip3 install scikit-learn nltk rouge-score

# Install retrieval packages
pip3 install faiss-cpu rank-bm25

# Install visualization packages
pip3 install matplotlib seaborn

# Install utilities
pip3 install click pyyaml

# Download NLTK data
python3 -c "
import nltk
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print('✓ NLTK data downloaded')
except:
    print('⚠ NLTK download failed')
"

# Check GPU availability
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA version: {torch.version.cuda}')
    print(f'✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('⚠ CUDA not available, using CPU')
"

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{chroniclingqa,temprageval,generated_questions,atlas_2021}
mkdir -p models/{cache,time_aware_contriever}
mkdir -p outputs/{chroniclingqa,temprageval,mrag}
mkdir -p logs

# Set permissions
chmod +x scripts/*.sh

echo ""
echo "✓ Colab setup completed!"
echo "✓ Ready to run: !bash scripts/run_complete_pipeline.sh"
echo ""
echo "🚀 To start the pipeline, run:"
echo "   !bash /content/TimeAwareRAG_Final/scripts/run_complete_pipeline.sh"