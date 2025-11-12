# Temporal-Retrieval-Augmented-Generation

This repository contains a re-implementation of MRAG (Multi-hop Retrieval-Augmented Generation) as a baseline system for temporal question answering. The project focuses on improving first-stage retrieval to be more time-aware, so that the initial candidate pool already respects temporal constraints, allowing MRAG to start from stronger candidates.

## Overview

The baseline system uses **Contriever** to retrieve top-K candidates from a fixed Wikipedia pool, then **MRAG** applies question-focused cues (keywords and optionally summaries) to re-rank those candidates before answer generation. The system is evaluated on **TempRAGEval** (n = 1,194) with metrics including Hit@K, MRR@K, and Recall@K, comparing vanilla Contriever retrieval against MRAG re-ranked retrieval. Results are contextualized within the broader temporal-QA landscape, including comparisons with TimeQA and SituatedQA.

## Repository Structure

### 📁 Baseline Implementation

This folder contains the core baseline implementation files:

- **`Custom ATLAS-2021 Corpus.ipynb`**: Creates a custom corpus for the MRAG baseline implementation. This notebook builds a "covered slice" from TempRAGEval gold evidence sentences by:
  1. Extracting normalized gold evidence sentences from TempRAGEval
  2. Using Aho-Corasick pattern matching to find Wikipedia pages containing these sentences
  3. Collecting all passages from matched pages as positive examples
  4. Augmenting the corpus with BM25-mined hard negative passages (targeting ~95% negatives)
  5. Producing a balanced corpus with approximately 219,940 passages (5% positives, 95% negatives) for training and evaluation

- **`MRAG Baseline Implementation.ipynb`**: Implements the complete MRAG baseline pipeline:
  1. **Retrieval Stage**: Uses Contriever to retrieve top-K candidates (default K=100) from the custom corpus using FAISS indexing
  2. **Re-ranking Stage**: Applies MRAG's question-focused re-ranking by:
     - Extracting and expanding keywords from normalized questions
     - Generating query-focused summaries (QFS) for top contexts using Phi-3.5-mini-instruct
     - Scoring sentences based on keyword matches with type-specific weights
     - Re-ranking passages by aggregated sentence scores
  3. **Answer Generation**: Uses the re-ranked contexts to generate answers via LLM
  4. **Evaluation**: Computes retrieval metrics (Recall@K, Hit@K, MRR@K, MAP@K, nDCG@K) comparing vanilla Contriever vs. MRAG re-ranked retrieval

### 📁 Proposed Methods - Explorations

This folder contains exploratory notebooks investigating various approaches to temporal-aware retrieval. The notebooks explore different techniques including Dense Passage Retrieval (DPR) with temporal embeddings, temporal hard negative mining strategies, and other experimental methods for incorporating temporal information into the retrieval pipeline. These explorations aim to improve upon the baseline by making the first-stage retrieval more time-aware.

### 📁 Corpus

Contains the corpus JSONL files used for retrieval:
- `text-list-100-sec.jsonl`: Wikipedia text passages
- `infobox.jsonl`: Wikipedia infobox data

These files are part of the ATLAS-2021 corpus and are processed by the `Custom ATLAS-2021 Corpus.ipynb` notebook to create the custom corpus for evaluation.

### 📁 Reports

Contains project documentation including the project proposal, midterm reports, and presentations.

## Key Components

### Retrieval Pipeline

1. **First-Stage Retrieval (Contriever)**: 
   - Encodes questions and passages using the `facebook/contriever-msmarco` model
   - Builds a FAISS index over the corpus for fast similarity search
   - Retrieves top-K candidates based on cosine similarity

2. **Second-Stage Re-ranking (MRAG)**:
   - Keyword extraction and expansion from normalized questions
   - Query-focused summarization for top contexts
   - Sentence-level keyword scoring with type-specific weights (special: 1.0, superlative: 0.7, general: 0.4, numeric: 0.5, adjective: 0.4)
   - Document-level aggregation and re-ranking

3. **Answer Generation**:
   - Uses Phi-3.5-mini-instruct to generate answers from re-ranked contexts
   - Applies prompt engineering to focus on temporal information

### Evaluation Metrics

The system evaluates retrieval performance using:
- **Hit@K**: Whether at least one relevant document appears in top-K
- **Recall@K**: Fraction of relevant documents retrieved in top-K
- **MRR@K**: Mean Reciprocal Rank of the first relevant document
- **MAP@K**: Mean Average Precision at K
- **nDCG@K**: Normalized Discounted Cumulative Gain at K

## Usage

1. **Create Custom Corpus**: Run `Baseline Implementation/Custom ATLAS-2021 Corpus.ipynb` to generate the corpus with hard negatives
2. **Run Baseline**: Execute `Baseline Implementation/MRAG Baseline Implementation.ipynb` to run the full MRAG pipeline and evaluation
3. **Explore Methods**: Review notebooks in `Proposed Methods - Explorations/` for temporal-aware retrieval experiments

## Dependencies

Key dependencies include:
- `transformers` (for Contriever and Phi-3.5 models)
- `faiss-cpu` (for efficient similarity search)
- `datasets` (for TempRAGEval dataset)
- `pyserini` (for BM25 negative mining)
- `nltk` (for text processing)
- `torch` (for model inference)

## Dataset

The system is evaluated on **TempRAGEval** (`siyue/TempRAGEval`), a temporal question answering dataset with 1,244 test examples. The dataset includes questions with various temporal relations (before, after, between, as of, etc.) and requires retrieving temporally relevant passages to answer questions correctly.

## Future Work

The goal is to improve the first-stage retrieval to be more time-aware, incorporating temporal constraints directly into the retrieval process rather than relying solely on re-ranking. This would allow MRAG to start from a stronger initial candidate pool that already respects temporal constraints.
