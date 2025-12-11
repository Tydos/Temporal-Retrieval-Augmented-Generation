#!/bin/bash

# Step 4: Evaluate on ChroniclingQA Dataset

echo "Evaluating models on ChroniclingQA dataset..."

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Use system Python (virtual environment will be added later if needed)
echo "Using Python: $(which python3)"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Check if trained model exists
if [ ! -d "contriever_finetuned_NEW_20k" ]; then
    echo "⚠ Warning: Time-aware model not found. Will evaluate base model only."
    echo "To train the time-aware model, run: ./scripts/03_train_contriever.sh"
fi

echo "Starting ChroniclingQA evaluation..."
echo "This will evaluate both base and time-aware Contriever models"
echo "Logs will be saved to: logs/chroniclingqa_evaluation.log"

# Create output and log directories
mkdir -p outputs/chroniclingqa
mkdir -p logs

# Run evaluation
python3 -u src/chroniclingqa_eval.py 2>&1 | tee logs/chroniclingqa_evaluation.log

# Check if results were generated
if [ -f "outputs/chroniclingqa_results.json" ]; then
    echo "✓ ChroniclingQA evaluation completed successfully!"
    
    # Display results summary
    echo ""
    echo "EVALUATION RESULTS SUMMARY:"
    echo "=========================="
    
    python3 -c "
import json
import pandas as pd

try:
    with open('outputs/chroniclingqa_results.json', 'r') as f:
        results = json.load(f)
    
    print('Results by model configuration:')
    print('-' * 50)
    
    for config, metrics in results.items():
        print(f'\\n{config.upper().replace(\"_\", \" \")}:')
        for metric, value in metrics.items():
            print(f'  {metric}: {value:.4f}')
    
    # Compare base vs time-aware on full dataset
    if 'base_model_full' in results and 'time_aware_model_full' in results:
        print('\\n' + '='*60)
        print('COMPARISON: Base vs Time-Aware (Full Dataset)')
        print('='*60)
        
        base_mrr = results['base_model_full'].get('mrr', 0)
        time_mrr = results['time_aware_model_full'].get('mrr', 0)
        improvement = ((time_mrr - base_mrr) / base_mrr * 100) if base_mrr > 0 else 0
        
        print(f'Base Model MRR: {base_mrr:.4f}')
        print(f'Time-Aware MRR: {time_mrr:.4f}')
        print(f'Improvement: {improvement:+.2f}%')
    
    # Check if CSV file exists
    import os
    if os.path.exists('outputs/chroniclingqa_results.csv'):
        print('\\n📊 Detailed results saved to: outputs/chroniclingqa_results.csv')
    
except Exception as e:
    print(f'Error reading results: {e}')
"
    
else
    echo "✗ Error: Evaluation failed!"
    echo "Check logs/chroniclingqa_evaluation.log for details"
    exit 1
fi

# Generate evaluation plots (optional)
echo ""
echo "Generating evaluation plots..."

python3 -c "
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os

try:
    with open('outputs/chroniclingqa_results.json', 'r') as f:
        results = json.load(f)
    
    # Create comparison plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Recall@K comparison
    models = []
    recall_1 = []
    recall_5 = []
    recall_10 = []
    
    for config, metrics in results.items():
        if 'full' in config:  # Only full dataset results
            models.append(config.replace('_', ' ').title())
            recall_1.append(metrics.get('recall@1', 0))
            recall_5.append(metrics.get('recall@5', 0))
            recall_10.append(metrics.get('recall@10', 0))
    
    if models:
        x = np.arange(len(models))
        width = 0.25
        
        ax1.bar(x - width, recall_1, width, label='Recall@1')
        ax1.bar(x, recall_5, width, label='Recall@5') 
        ax1.bar(x + width, recall_10, width, label='Recall@10')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Recall')
        ax1.set_title('Recall Comparison (ChroniclingQA Full)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: MRR comparison
    mrr_scores = []
    mrr_labels = []
    
    for config, metrics in results.items():
        mrr_labels.append(config.replace('_', ' ').title())
        mrr_scores.append(metrics.get('mrr', 0))
    
    if mrr_scores:
        bars = ax2.bar(mrr_labels, mrr_scores, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'][:len(mrr_scores)])
        ax2.set_xlabel('Model Configuration')
        ax2.set_ylabel('MRR Score')
        ax2.set_title('Mean Reciprocal Rank (MRR) Comparison')
        ax2.set_xticklabels(mrr_labels, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, mrr_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('outputs/chroniclingqa_evaluation_plots.png', dpi=300, bbox_inches='tight')
    print('📊 Evaluation plots saved to: outputs/chroniclingqa_evaluation_plots.png')
    
except Exception as e:
    print(f'Note: Could not generate plots: {e}')
"

echo ""
echo "ChroniclingQA evaluation completed!"
echo "Results saved to: outputs/chroniclingqa_results.json"
echo "CSV format: outputs/chroniclingqa_results.csv"
echo ""
echo "Next step: Run ./scripts/05_mrag_temprageval.sh"