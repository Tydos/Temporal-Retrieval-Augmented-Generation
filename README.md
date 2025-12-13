# Time-Aware RAG: Temporal Question Answering with Fine-tuned Contriever

A comprehensive research project implementing time-aware Retrieval-Augmented Generation (RAG) using fine-tuned Contriever models. This system generates temporal questions with T5, fine-tunes Contriever for temporal awareness, and evaluates performance using multi-hop retrieval (MRAG) on ChroniclingQA and SQuAD datasets.

## Project overview

Objectives
- Fine-tune Contriever on T5-generated temporal questions
- Compare base and time-aware models
- Implement MRAG (multi-hop retrieval) and evaluate its benefit
- Evaluate on T5-generated in-domain test set, ChroniclingQA (out-of-domain), and SQuAD (temporal subset)

Key components
- T5 question generation (in-domain sample generation)
- Time-aware Contriever training (temporal embeddings)
- MRAG multi-hop retrieval and fusion
- Evaluation scripts for ChroniclingQA, MRAG, and SQuAD-filtered runs

## Requirements

System
- Python 3.8+
- 16GB+ RAM recommended
- GPU with CUDA recommended for training

Dependencies
All Python dependencies are listed in `requirements.txt`. Install with:

```bash
pip install -r requirements.txt
```

## How to run it?

Make scripts executable and run the pipeline or the step-by-step scripts in `scripts/`.

Examples — two ways to run

1) Run the full pipeline with the wrapper script (recommended for end-to-end reproducibility):

```bash
chmod +x scripts/run_complete_pipeline.sh
./scripts/run_complete_pipeline.sh
```

2) Run the pipeline step-by-step by executing scripts `01` through `06` manually (order matters):

```bash
chmod +x scripts/*.sh
./scripts/01_setup_environment.sh    # prepares FineWeb and required dependencies (this step is required)
./scripts/02_generate_questions.sh  # T5 question generation -> data/generated_questions/
./scripts/03_train_contriever.sh    # train time-aware Contriever (run if comparing base vs time-aware)
./scripts/05_mrag_caqa_eval.sh      # MRAG integration and CAQA/ChroniclingQA MRAG evaluation (required; must run before SQuAD runs)
./scripts/04_evaluate_chroniclingqa.sh  # T5 in-domain (if available) + ChroniclingQA OOD evaluation
./scripts/06_mrag_squad_eval.sh     # SQuAD temporal-filtered evaluation + MRAG (runs after MRAG CAQA)
```

Notes
- FineWeb preparation (`01_setup_environment.sh`) is required for this pipeline; do not skip it — the system depends on `data/fineweb/fineweb_passages.json` and the generated indices.
- MRAG CAQA step (`05_mrag_caqa_eval.sh`) is mandatory and must be executed before the SQuAD MRAG evaluation.

## Exact script -> code mapping (what each numbered script does)

- `scripts/01_setup_environment.sh`
  - Installs Python deps, sets up directories, and (by default) runs `src/fineweb_loader.py` to prepare FineWeb passages.

- `scripts/02_generate_questions.sh` -> `src/question_generation.py`
  - Uses a T5 generator to produce temporal question-passage pairs and writes `data/generated_questions/sample_generated_questions.json`.
  - If `data/fineweb/fineweb_passages.json` exists, it samples passages from FineWeb; otherwise it falls back to a small set of hard-coded example passages.

- `scripts/03_train_contriever.sh` -> `src/contriever_training.py`
  - Fine-tunes a time-aware Contriever and saves to `contriever_finetuned_NEW_20k/` or `models/time_aware_contriever/` depending on configuration.
  - If not run, evaluation scripts will fall back to the base Contriever model.

- `scripts/04_evaluate_chroniclingqa.sh` -> `src/chroniclingqa_eval.py`
  - Runs two main evaluations:
    1. T5 in-domain test set evaluation (if the T5 test mapping is found) and
    2. ChroniclingQA (out-of-domain) evaluation using the `Bhawna/ChroniclingAmericaQA` dataset via `datasets`.
  - Produces `outputs/chroniclingqa_results.json` and `outputs/chroniclingqa_results.csv` and optional plots.
  - Important: by default `load_t5_test_data()` expects `data/fineweb/fineweb_passages.json` to map `sample_{id}` passage ids to original passages. If FineWeb is not available, T5 evaluation may be skipped unless the code uses the generated file directly (see Recommendations).

- `scripts/05_mrag_caqa_eval.sh` -> `src/mrag_integration.py`
  - Runs MRAG integration and evaluation on ChroniclingQA/CAQA-style runs. Uses precomputed window embeddings and optimized MaxSim reranking.

- `scripts/06_mrag_squad_eval.sh` -> `src/squad_time_filter_eval.py`
  - Loads SQuAD validation, filters for temporal questions (year regex and 'when' filters), builds indices, and evaluates four configurations: base, base+MRAG, time-aware, time-aware+MRAG.
  - Output: `outputs/archivalqa_filtered_results.json` (and console summary). There may also be `outputs/squad_filtered_results.json` from prior runs.

<!-- 'Run the full pipeline' merged into Quick start above -->

## Results (summary drawn from `outputs/` at time of update)

I read the available result files in `outputs/` to produce this concise summary. The raw JSON files live in `outputs/`.

1) ChroniclingQA (T5 in-domain summary)
- Source: `outputs/chroniclingqa_results.json`
- Representative (selected keys):
  - T5 In-Domain Full [BASE]: hit@1 = 0.784, hit@5 = 0.87433, hit@10 = 0.9
  - T5 In-Domain Full [TIMEAWARE]: hit@1 = 0.849, hit@5 = 0.92233, hit@10 = 0.94233

2) ChroniclingQA (CAQA-style / out-of-domain)
- Source: `outputs/chroniclingqa_results.json`
  - CAQA Out-of-Domain Full [BASE]: hit@1 = 0.47816, hit@5 = 0.65622, hit@10 = 0.71648
  - CAQA Out-of-Domain Full [TIMEAWARE]: hit@1 = 0.50355, hit@5 = 0.67994, hit@10 = 0.73962

3) MRAG evaluation
- Source: `outputs/mrag_eval_results_fast.json` (CAQA+MRAG implementation)
  - CAQA+MRAG (base / time-aware variants): see the Results table below for Hit@K and MRR values drawn from this file.

4) SQuAD-filtered (temporal) + MRAG
- Source: `outputs/squad_filtered_results.json` or `outputs/archivalqa_filtered_results.json`
  - base_only: hit@1 = 0.717, hit@5 = 0.921
  - mrag_base: hit@1 = 0.783, hit@5 = 0.943
  - mrag_time_aware: hit@1 = 0.785, hit@5 = 0.942

Full raw outputs (files present)
- `outputs/chroniclingqa_results.json` — ChroniclingQA and CAQA-style metrics
- `outputs/chroniclingqa_results.csv` — CSV of ChroniclingQA results
- `outputs/mrag_eval_results_fast.json` — CAQA+MRAG (fast) evaluation
- `outputs/squad_filtered_results.json` — SQuAD-filtered evaluation

If you want these numbers presented as plots embedded into the README, I can generate PNGs from the JSON and add them under `outputs/` and reference them here.

### Detailed results (full outputs)

Below are full metric tables taken directly from the JSON output files in `outputs/`. These include Hit@K, MRR@K and Recall@K where available — no metrics omitted.

#### ChroniclingQA results (`outputs/chroniclingqa_results.json`)

Below are the T5 in-domain ChroniclingQA results split into Full and Year-subset tables.

T5 In-Domain — Full

| Variant | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@1 | MRR@5 | MRR@10 | MRR@20 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.78400 | 0.87433 | 0.90000 | 0.92033 | 0.78400 | 0.81966 | 0.82313 | 0.82456 | 0.78400 | 0.87433 | 0.90000 |
| timeaware | **0.84900** | **0.92233** | **0.94233** | **0.95600** | **0.84900** | **0.87926** | **0.88185** | **0.88283** | **0.84900** | **0.92233** | **0.94233** |

T5 In-Domain — Year-subset

| Variant | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@1 | MRR@5 | MRR@10 | MRR@20 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.63895 | 0.75862 | 0.79716 | 0.83976 | 0.63895 | 0.68310 | 0.68814 | 0.69097 | 0.63895 | 0.75862 | 0.79716 |
| timeaware | **0.76065** | **0.87221** | **0.90872** | **0.92901** | **0.76065** | **0.80510** | **0.81005** | **0.81150** | **0.76065** | **0.87221** | **0.90872** |

#### CAQA + MRAG results (`outputs/mrag_eval_results_fast.json`)

| Eval key | Hit@1 | Hit@5 | Hit@10 | Hit@20 | Hit@50 | MRR@1 | MRR@5 | MRR@10 | MRR@20 | MRR@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_only | 0.40361 | 0.57178 | 0.63823 | 0.70057 | 0.78097 | 0.40361 | 0.46812 | 0.47692 | 0.48115 | 0.48373 |
| time_aware_only | 0.47826 | 0.66694 | 0.71452 | 0.77933 | 0.83347 | 0.47826 | 0.55127 | 0.55782 | 0.56226 | 0.56409 |
| mrag_base | 0.57260 | 0.73503 | 0.76784 | 0.79245 | 0.80968 | 0.57260 | 0.63749 | 0.64197 | 0.64376 | 0.64435 |
| mrag_time_aware | **0.59147** | **0.75308** | **0.79327** | **0.82445** | **0.84660** | **0.59147** | **0.65537** | **0.66107** | **0.66341** | **0.66410** |

#### SQuAD-filtered results (`outputs/squad_filtered_results.json`)

| Eval key | Hit@1 | Hit@5 | Hit@10 | Hit@20 | MRR@1 | MRR@5 | MRR@10 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base_only | 0.71700 | 0.92100 | 0.95400 | 0.97200 | 0.71700 | 0.79925 | 0.80381 | 0.80508 |
| mrag_base | 0.78300 | **0.94300** | 0.95900 | 0.97500 | 0.78300 | 0.84748 | 0.84977 | 0.85096 |
| time_aware_only | 0.72600 | 0.92600 | 0.95700 | **0.98400** | 0.72600 | 0.80647 | 0.81078 | 0.81270 |
| mrag_time_aware | **0.78500** | 0.94200 | **0.96400** | 0.97800 | **0.78500** | **0.85092** | **0.85385** | **0.85489** |

Raw JSON files used to produce these tables are included in `outputs/`:

- `outputs/chroniclingqa_results.json`
- `outputs/mrag_eval_results_fast.json` (CAQA+MRAG implementation)
- `outputs/squad_filtered_results.json`

## Reproducibility 

If you only want to run T5 test set (in-domain), ChroniclingQA (OOD), and SQuAD (temporal)+MRAG evaluations, the minimal sequence is:

1. Install dependencies (skip FineWeb download manually if you prefer):
```bash
pip install -r requirements.txt
# optionally skip the FineWeb step in scripts/01_setup_environment.sh
```

2. Generate T5 questions (uses fallback samples if FineWeb is not present):
```bash
./scripts/02_generate_questions.sh
```

3. (Optional) Train time-aware Contriever if you want time-aware comparisons:
```bash
./scripts/03_train_contriever.sh
```

4. Evaluate T5 in-domain and ChroniclingQA OOD:
```bash
./scripts/04_evaluate_chroniclingqa.sh
```

5. Evaluate SQuAD temporal subset + MRAG:
```bash
./scripts/06_mrag_squad_eval.sh
```

6. Optional: MRAG CAQA run (if you want a separate MRAG pass on ChroniclingQA):
```bash
./scripts/05_mrag_caqa_eval.sh
```

## Where the important files are

- Generated questions (T5): `data/generated_questions/sample_generated_questions.json`
- FineWeb (if downloaded): `data/fineweb/fineweb_passages.json`
- Time-aware model (if trained): `contriever_finetuned_NEW_20k/` or `models/time_aware_contriever/`
- Index: `contriever_mining_index_fineweb_20k/mining.index`
- Outputs (evaluation results): `outputs/*.json`, `outputs/*.csv`, `outputs/*.png`


### Weights & Biases Integration
```bash
# Setup wandb logging
pip install wandb
wandb login

# Set environment variable
export WANDB_API_KEY="your_api_key"
export REPORT_TO="wandb"
```

## Monitoring Progress

### Real-time Monitoring
```bash
# Monitor training progress
tail -f logs/contriever_training.log

# Monitor evaluation
tail -f logs/chroniclingqa_evaluation.log
```

### Checking Results
```bash
# View latest results
cat outputs/chroniclingqa_results.json

# Check model status
ls -la models/time_aware_contriever/
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in `configs/config.yaml`
   - Use CPU-only mode: `export CUDA_VISIBLE_DEVICES=""`

2. **Missing Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Permission Errors**
   ```bash
   chmod +x scripts/*.sh
   ```

4. **Model Loading Issues**
   - Clear model cache: `rm -rf models/cache/`
   - Check available disk space

### Log Files
Check these logs for detailed error information:
- `logs/question_generation.log`
- `logs/contriever_training.log`
- `logs/chroniclingqa_evaluation.log`
- `logs/mrag_temprageval.log`

## Methodology

### 1. Question Generation
- Uses T5-base to generate temporal questions
- Focuses on time-related aspects of passages
- Creates question-passage pairs for training

### 2. Model Architecture
- Base: Facebook Contriever encoder
- Enhancement: Temporal attention mechanism
- Training: Contrastive learning on temporal pairs

### 3. Evaluation Protocol
- **ChroniclingQA**: Full dataset and temporal subset
- **SQuAD**: Temporal subset evaluation using SQuAD validation split
- **MRAG**: Multi-hop retrieval comparison

### 4. Metrics
- **MRR**: Mean Reciprocal Rank
- **Recall@K**: Retrieval recall at different K values
- **Precision@K**: Retrieval precision
