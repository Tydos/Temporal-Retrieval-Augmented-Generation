#!/bin/bash

# Step 2: Generate Temporal Questions using T5

echo "Generating temporal questions with T5 model..."

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use system Python (virtual environment will be added later if needed)
echo "Using Python: $(which python3)"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Create log directory
mkdir -p logs

echo "Starting T5 question generation..."
echo "Logs will be saved to: logs/question_generation.log"

# Run question generation
python3 -u src/question_generation.py 2>&1 | tee logs/question_generation.log

# Check if output was generated
if [ -f "data/generated_questions/sample_generated_questions.json" ]; then
    echo "✓ Question generation completed successfully!"
    
    # Show statistics
    python3 -c "
import json
with open('data/generated_questions/sample_generated_questions.json', 'r') as f:
    data = json.load(f)
print(f'Generated {len(data)} question-passage pairs')
print(f'Sample question: {data[0][\"question\"]}')
print(f'Sample passage: {data[0][\"passage\"][:100]}...')
"
else
    echo "✗ Error: Question generation failed!"
    echo "Check logs/question_generation.log for details"
    exit 1
fi

# Optional: Generate questions from custom dataset
echo ""
echo "To generate questions from your own dataset:"
echo "1. Place your data in JSON format in data/ directory"
echo "2. Update the dataset path in configs/config.yaml"
echo "3. Modify src/question_generation.py to load your dataset"
echo ""

echo "Question generation completed!"
echo "Generated data saved to: data/generated_questions/"
echo ""
echo "Next step: Run ./scripts/03_train_contriever.sh"