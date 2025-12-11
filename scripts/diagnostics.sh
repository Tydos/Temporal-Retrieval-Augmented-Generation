#!/bin/bash

# Diagnostic script to identify environment issues

echo "=== TIME-AWARE RAG DIAGNOSTICS ==="
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Shell: $SHELL"
echo ""

echo "=== DIRECTORY INFORMATION ==="
echo "Current working directory: $(pwd)"
echo "Script location: ${BASH_SOURCE[0]}"
echo "Script directory: $(dirname "${BASH_SOURCE[0]}")"
echo ""

echo "=== PYTHON INFORMATION ==="
which python3
python3 --version
echo ""

echo "=== PROJECT STRUCTURE CHECK ==="
if [ -f "requirements.txt" ]; then
    echo "✓ requirements.txt found"
else
    echo "✗ requirements.txt NOT found"
fi

if [ -f "configs/config.yaml" ]; then
    echo "✓ configs/config.yaml found"
else
    echo "✗ configs/config.yaml NOT found"
fi

if [ -d "src" ]; then
    echo "✓ src directory found"
    echo "  Files in src: $(ls src/ 2>/dev/null | wc -l)"
else
    echo "✗ src directory NOT found"
fi

echo ""
echo "=== VIRTUAL ENVIRONMENT TEST ==="
if command -v python3 >/dev/null 2>&1; then
    echo "✓ python3 available"
    
    # Test venv creation
    echo "Testing virtual environment creation..."
    rm -rf test_venv_diagnostic
    
    if python3 -m venv test_venv_diagnostic; then
        echo "✓ Virtual environment creation successful"
        
        if [ -f "test_venv_diagnostic/bin/activate" ]; then
            echo "✓ Activation script exists"
            
            # Test activation
            if source test_venv_diagnostic/bin/activate; then
                echo "✓ Virtual environment activation successful"
                echo "  Virtual env Python: $(which python)"
                deactivate 2>/dev/null || true
            else
                echo "✗ Virtual environment activation failed"
            fi
        else
            echo "✗ Activation script not found"
        fi
        
        # Cleanup
        rm -rf test_venv_diagnostic
    else
        echo "✗ Virtual environment creation failed"
    fi
else
    echo "✗ python3 not available"
fi

echo ""
echo "=== PERMISSIONS CHECK ==="
echo "Current user: $(id -un)"
echo "Directory permissions: $(ls -ld . 2>/dev/null)"
echo "Can create files: $(touch test_file_permissions 2>/dev/null && echo "Yes" && rm test_file_permissions || echo "No")"

echo ""
echo "=== RECOMMENDATIONS ==="

if [ ! -f "requirements.txt" ]; then
    echo "⚠ You don't seem to be in the correct project directory"
    echo "  Run: cd /Users/Patron/Downloads/TimeAwareRAG_Final"
fi

echo "To run the pipeline safely:"
echo "1. cd /Users/Patron/Downloads/TimeAwareRAG_Final"
echo "2. bash scripts/run_complete_pipeline.sh"
echo ""
echo "For step-by-step execution:"
echo "1. bash scripts/01_setup_environment.sh"
echo "2. bash scripts/02_generate_questions.sh" 
echo "... (and so on)"

echo ""
echo "=== END DIAGNOSTICS ==="