"""
Full SQuAD Time Filtered Evaluation
Evaluates 4 configurations on STRICTLY temporal questions from SQuAD:
1. Base Model Only
2. Base Model + MRAG
3. Time-Aware Model Only
4. Time-Aware Model + MRAG
"""

import os
import re
import json
import yaml
import torch
import logging
import numpy as np
import faiss
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from mrag_integration import (
    mrag_rerank_1, 
    precompute_window_embeddings, 
    build_faiss_index, 
    encode_texts
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RETRIEVE_TOPK = 100
EVAL_KS = (1, 5, 10, 20)

# --- REGEX ADAPTER (Crucial for unmodified MRAG) ---
def adapt_question_for_mrag_regex(question: str) -> str:
    q_clean = question.strip()
    
    # 1. Check existing matches
    if re.search(r'(between|from)\s+.*?\d{4}\s+(and|to|-)\s+.*?\d{4}', q_clean, re.IGNORECASE):
        return q_clean
    if re.search(r'(as of|in|on|around|by)\s+.*?\d{4}', q_clean, re.IGNORECASE):
        return q_clean

    # 2. Inject patterns for implicit years
    range_match = re.search(r'\b(\d{4})[-–](\d{4})\b', q_clean)
    if range_match:
        y1, y2 = range_match.groups()
        return f"{q_clean} from {y1} to {y2}"

    year_matches = re.findall(r'\b(1[0-9]{3}|20[0-2][0-9])\b', q_clean)
    if year_matches:
        target_year = year_matches[-1]
        return f"{q_clean} in {target_year}"
            
    return q_clean

def evaluate_configuration(
    index,
    model,
    tokenizer,
    questions,
    gold_indices,
    passages_list,
    use_mrag=False,
    desc="",
    window_emb_tensor=None,
    doc_window_map=None
):
    print(f"\n[EVAL] {desc} | q={len(questions)} | mrag={use_mrag}")
    
    # 1. Retrieval
    q_embs = encode_texts(model, tokenizer, questions)
    scores, ids = index.search(q_embs, RETRIEVE_TOPK)
    
    metrics = {f"hit@{k}": 0.0 for k in EVAL_KS}
    metrics.update({f"mrr@{k}": 0.0 for k in EVAL_KS})
    
    for i, q_text in enumerate(tqdm(questions, desc="Reranking")):
        gold_idx = gold_indices[i]
        
        cand_indices = ids[i]
        cand_scores_raw = scores[i]
        
        # Valid candidates
        valid_cand_indices = [int(idx) for idx in cand_indices if idx >= 0]
        cand_texts = [passages_list[idx] for idx in valid_cand_indices]

        if use_mrag:
            adapted_q = adapt_question_for_mrag_regex(q_text)
            
            # Pass GLOBAL IDs (valid_cand_indices) to match doc_window_map
            final_ranked_ids = mrag_rerank_1(
                adapted_q, 
                cand_texts, 
                valid_cand_indices, 
                model, 
                tokenizer,
                base_scores=cand_scores_raw,
                blend_weight=0.0,
                temporal_weight=0.9,
                window_emb_tensor=window_emb_tensor,
                doc_window_map=doc_window_map
            )
            
            # Fallback
            if len(final_ranked_ids) < len(valid_cand_indices):
                missing = [pid for pid in valid_cand_indices if pid not in final_ranked_ids]
                final_ranked_ids.extend(missing)
        else:
            final_ranked_ids = valid_cand_indices

        # Compute Metrics
        for k in EVAL_KS:
            if gold_idx in final_ranked_ids[:k]:
                metrics[f"hit@{k}"] += 1.0
                for rank, pid in enumerate(final_ranked_ids[:k], start=1):
                    if pid == gold_idx:
                        metrics[f"mrr@{k}"] += 1.0 / rank
                        break

    total = len(questions)
    metrics = {k: v / total for k, v in metrics.items()}
    print(f"Results for {desc}: {metrics}")
    return metrics

def main():
    # Load Config
    try:
        with open('configs/config.yaml', 'r') as f: config = yaml.safe_load(f)
        base_name = config['models']['base_contriever']['name']
        time_path = config['models']['time_aware_contriever']['output_dir']
    except:
        base_name = "facebook/contriever-msmarco"
        time_path = "./models/time_aware_contriever"

    print(f"Base: {base_name}")
    print(f"Time-Aware: {time_path}")

    # Load Models
    print("Loading Models...")
    base_tok = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModel.from_pretrained(base_name).to(DEVICE).eval()
    
    try:
        time_model = AutoModel.from_pretrained(time_path).to(DEVICE).eval()
        time_tok = AutoTokenizer.from_pretrained(base_name)
    except:
        print("WARNING: Time-Aware model not found. Using Base model for both.")
        time_model = base_model
        time_tok = base_tok

    # --- Load SQuAD ONLY ---
    print("Loading SQuAD dataset (validation split)...")
    dataset = load_dataset("squad", split="validation")

    passages = []
    questions = []
    gold_indices = []
    seen_passages = {}
    
    # FILTER: Regex for years (1700-2029) OR strict "When" questions
    year_re = re.compile(r"\b(17|18|19|20)\d{2}\b")
    when_re = re.compile(r"^(when|what year|in what year)", re.IGNORECASE)
    
    print("Filtering for strictly TEMPORAL questions (Years or 'When')...")
    scanned_count = 0
    
    for row in tqdm(dataset, desc="Scanning"):
        scanned_count += 1
        
        # SQuAD keys
        q = row['question']
        ctx = row['context']
        
        # CRITICAL FILTER: Keep if Year found OR starts with "When"
        if not (year_re.search(q) or when_re.search(q)):
            continue
            
        if ctx not in seen_passages:
            seen_passages[ctx] = len(passages)
            passages.append(ctx)
            
        questions.append(q)
        gold_indices.append(seen_passages[ctx])
        
        # Limit to 1000 temporal questions for reasonable runtime
        if len(questions) >= 1000: 
            break

    print(f"\n[Stats] Scanned: {scanned_count} | Kept (Temporal): {len(questions)}")
    print(f"[Stats] Unique Passages: {len(passages)}")
    
    # --- PRINT SAMPLES ---
    print("\n" + "="*40)
    print("SAMPLE FILTERED QUESTIONS")
    print("="*40)
    for i in range(min(5, len(questions))):
        print(f"Q{i+1}: {questions[i]}")
    print("="*40 + "\n")

    # ==========================================
    # PHASE 1: BASE MODEL
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 1: BASE MODEL EVALUATION")
    print("="*40)
    
    base_index = build_faiss_index(base_model, base_tok, passages)
    print("Pre-computing Base Windows...")
    base_win_tensor, base_win_map = precompute_window_embeddings(base_model, base_tok, passages)
    
    # 1. Base Only
    res_base = evaluate_configuration(
        base_index, base_model, base_tok, 
        questions, gold_indices, passages, 
        use_mrag=False, desc="Base Only"
    )
    
    # 2. MRAG + Base
    res_mrag_base = evaluate_configuration(
        base_index, base_model, base_tok, 
        questions, gold_indices, passages, 
        use_mrag=True, desc="MRAG + Base",
        window_emb_tensor=base_win_tensor,
        doc_window_map=base_win_map
    )

    # ==========================================
    # PHASE 2: TIME-AWARE MODEL
    # ==========================================
    print("\n" + "="*40)
    print("PHASE 2: TIME-AWARE MODEL EVALUATION")
    print("="*40)
    
    time_index = build_faiss_index(time_model, time_tok, passages)
    print("Pre-computing Time-Aware Windows...")
    time_win_tensor, time_win_map = precompute_window_embeddings(time_model, time_tok, passages)
    
    # 3. Time-Aware Only
    res_time = evaluate_configuration(
        time_index, time_model, time_tok, 
        questions, gold_indices, passages, 
        use_mrag=False, desc="Time-Aware Only"
    )
    
    # 4. MRAG + Time-Aware
    res_mrag_time = evaluate_configuration(
        time_index, time_model, time_tok, 
        questions, gold_indices, passages, 
        use_mrag=True, desc="MRAG + Time-Aware",
        window_emb_tensor=time_win_tensor,
        doc_window_map=time_win_map
    )

    # ==========================================
    # SUMMARY
    # ==========================================
    
    print("\n" + "="*90)
    print(f"{'FINAL RESULTS':^90}")
    print("="*90)
    
    # Headers
    headers = ["Configuration", "Hit@1", "Hit@5", "Hit@10", "Hit@20", "MRR@1", "MRR@5", "MRR@10", "MRR@20"]
    row_fmt = "{:<22} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7} | {:<7}"
    print(row_fmt.format(*headers))
    print("-" * 90)
    
    def print_row(name, r):
        print(row_fmt.format(
            name,
            f"{r['hit@1']:.4f}", f"{r['hit@5']:.4f}", f"{r['hit@10']:.4f}", f"{r['hit@20']:.4f}",
            f"{r['mrr@1']:.4f}", f"{r['mrr@5']:.4f}", f"{r['mrr@10']:.4f}", f"{r['mrr@20']:.4f}"
        ))
        
    print_row("Base Only", res_base)
    print_row("Base + MRAG", res_mrag_base)
    print_row("Time-Aware Only", res_time)
    print_row("Time-Aware + MRAG", res_mrag_time)
    print("="*90)

    # Save to file
    all_results = {
        "base_only": res_base,
        "mrag_base": res_mrag_base,
        "time_aware_only": res_time,
        "mrag_time_aware": res_mrag_time
    }
    with open("outputs/archivalqa_filtered_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()