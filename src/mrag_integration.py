"""
MRAG (Multi-hop Retrieval-Augmented Generation) Integration - OPTIMIZED
Optimized with Pre-computed Window Embeddings to remove redundant encoding.
"""

import os
import re
import json
import yaml
import torch
import logging
import numpy as np
import faiss
import nltk
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
ENCODING_BATCH_SIZE = 128  # Increased for speed
MAX_LENGTH = 256
RETRIEVE_TOPK = 100
EVAL_KS = (1, 5, 10, 20, 50)

# Ensure NLTK resources
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# --- Regex for Temporal Constraints ---
TEMP_RANGE_RE = re.compile(
    r"(between|from)\s+(\d{4})\s+(and|to|-)\s+(\d{4})",
    flags=re.IGNORECASE,
)
TEMP_POINT_RE = re.compile(
    r"(as of|in|on|around|by)\s+(\d{4})",
    flags=re.IGNORECASE,
)
YEAR_ANY_RE = re.compile(
    r"\b(1[5-9]\d{2}|20\d{2}|2100)\b"
)
YEAR_PATTERN = re.compile(r"\b(1[5-9]\d{2}|20\d{2}|2100)\b")

# --- Helper Classes & Functions ---

@dataclass
class TemporalConstraintV1:
    type: str
    year: Optional[int] = None
    start_year: Optional[int] = None
    end_year: Optional[int] = None

def mean_pooling(last_hidden_state, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

@torch.no_grad()
def encode_texts(model, tokenizer, texts: List[str], batch_size: int = ENCODING_BATCH_SIZE, max_len: int = MAX_LENGTH):
    model.eval()
    all_vecs = []
    if not texts:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
        
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[i:i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(DEVICE)
        with torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out = model(**toks)
            pooled = mean_pooling(out.last_hidden_state, toks['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        all_vecs.append(pooled.cpu())
    
    return torch.cat(all_vecs, dim=0).numpy().astype(np.float32)

def build_faiss_index(model, tokenizer, passages: List[str]):
    print(f"Building FAISS index (IP) with {len(passages)} passages...")
    embs = encode_texts(model, tokenizer, passages)
    dim = embs.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    ids = np.arange(len(passages), dtype=np.int64)
    index.add_with_ids(embs, ids)
    return index

# --- Pre-computation Optimization ---

PRETOKENIZED_WINDOWS = {}
WINDOW_SIZE = 3
WINDOW_STRIDE = 1

def _get_doc_id(pid):
    return str(pid)

def pretokenize_passages(passages: List[str], ids: List[int]):
    """Splits passages into overlapping WINDOWS."""
    global PRETOKENIZED_WINDOWS
    
    for pid, text in zip(ids, passages):
        pid_str = _get_doc_id(pid)
        if pid_str in PRETOKENIZED_WINDOWS:
            continue
            
        try:
            snts = nltk.sent_tokenize(text or "")
        except Exception:
            snts = re.split(r"(?<=[.!?])\s+", text or "")
        snts = [s.strip() for s in snts if s.strip()]

        windows = []
        if not snts:
            windows = [""]
        else:
            if len(snts) <= WINDOW_SIZE:
                windows.append(" ".join(snts))
            else:
                for i in range(0, len(snts) - WINDOW_SIZE + 1, WINDOW_STRIDE):
                    window_text = " ".join(snts[i : i + WINDOW_SIZE])
                    windows.append(window_text)

        PRETOKENIZED_WINDOWS[pid_str] = windows

def precompute_window_embeddings(model, tokenizer, passages: List[str], batch_size: int = 128):
    """
    Pre-computes embeddings for all windows of all passages.
    Returns: 
       - full_tensor: Tensor of shape (Total_Windows, Hidden_Dim)
       - doc_window_map: Dict {doc_id_str: (start_index, count)}
    """
    print(f"Pre-computing window embeddings for {len(passages)} passages...")
    model.eval()
    
    # 1. Tokenize all passages into windows first
    all_window_texts = []
    doc_window_map = {} 
    
    current_idx = 0
    passage_ids = list(range(len(passages)))
    
    # Ensure windows are generated
    pretokenize_passages(passages, passage_ids)
    
    for pid in passage_ids:
        pid_str = _get_doc_id(pid)
        windows = PRETOKENIZED_WINDOWS.get(pid_str, [""])
        doc_window_map[pid_str] = (current_idx, len(windows))
        all_window_texts.extend(windows)
        current_idx += len(windows)
        
    print(f"Total windows to encode: {len(all_window_texts)}")
    
    # 2. Encode all windows
    all_embs_list = []
    
    for i in tqdm(range(0, len(all_window_texts), batch_size), desc="Encoding Windows"):
        batch = all_window_texts[i : i + batch_size]
        toks = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
            out = model(**toks)
            pooled = mean_pooling(out.last_hidden_state, toks['attention_mask'])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_embs_list.append(pooled.cpu()) 
            
    full_tensor = torch.cat(all_embs_list, dim=0)
    return full_tensor, doc_window_map

# --- MRAG Specific Logic ---

def decompose_question_temporal_v1(question_text: str):
    q = question_text.strip()
    m = TEMP_RANGE_RE.search(q)
    if m:
        y1, y2 = int(m.group(2)), int(m.group(4))
        start, end = min(y1, y2), max(y1, y2)
        tc = TemporalConstraintV1(type="range", start_year=start, end_year=end)
        mc = (q[:m.start()] + " " + q[m.end():]).strip()
        return (re.sub(r"\s+", " ", mc) if mc else q), tc

    m = TEMP_POINT_RE.search(q)
    if m:
        y = int(m.group(2))
        tc = TemporalConstraintV1(type="point", year=y)
        mc = (q[:m.start()] + " " + q[m.end():]).strip()
        return (re.sub(r"\s+", " ", mc) if mc else q), tc

    m = YEAR_ANY_RE.search(q)
    if m:
        y = int(m.group(1))
        tc = TemporalConstraintV1(type="point", year=y)
        mc = (q[:m.start()] + " " + q[m.end():]).strip()
        return (re.sub(r"\s+", " ", mc) if mc else q), tc

    return q, None

def extract_mc(q: str) -> str:
    mc_text, _ = decompose_question_temporal_v1(q)
    return mc_text

def get_dense_maxsim_scores_fast(
    question_text: str, 
    candidate_ids: List[int], 
    model, 
    tokenizer, 
    window_emb_tensor, 
    doc_window_map
) -> List[float]:
    """Optimized MaxSim using pre-computed embeddings."""
    
    # 1. Encode Question (On the fly)
    model.eval()
    q_tok = tokenizer([question_text], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad(), torch.autocast(DEVICE, dtype=AMP_DTYPE, enabled=torch.cuda.is_available()):
        q_out = model(**q_tok)
        q_emb = mean_pooling(q_out.last_hidden_state, q_tok['attention_mask'])
        q_emb = torch.nn.functional.normalize(q_emb, p=2, dim=1) # (1, H)

    # 2. Gather relevant window embeddings
    relevant_indices = []
    doc_boundaries = [] # (start_in_subset, end_in_subset)
    
    curr_subset_idx = 0
    for cid in candidate_ids:
        start_idx, count = doc_window_map.get(str(cid), (0, 0))
        # Add range of indices
        relevant_indices.extend(range(start_idx, start_idx + count))
        doc_boundaries.append((curr_subset_idx, curr_subset_idx + count))
        curr_subset_idx += count

    if not relevant_indices:
        return [0.0] * len(candidate_ids)

    # 3. Compute Similarity (Matrix Multiplication)
    # Move specific embeddings to GPU
    relevant_indices_tensor = torch.tensor(relevant_indices, dtype=torch.long)
    # select rows from CPU tensor, move to GPU
    subset_embs = window_emb_tensor.index_select(0, relevant_indices_tensor).to(DEVICE) 
    
    sims = torch.mm(subset_embs, q_emb.T).squeeze(1) # (N_windows,)
    sims_cpu = sims.float().cpu().numpy()

    # 4. Max Pool per document
    doc_max_scores = []
    for start, end in doc_boundaries:
        if start < end:
            doc_max_scores.append(float(sims_cpu[start:end].max()))
        else:
            doc_max_scores.append(0.0)
            
    return doc_max_scores

DOC_YEARS_CACHE = {}

def get_doc_years(doc_id: int, doc_text: str) -> List[int]:
    if doc_id in DOC_YEARS_CACHE:
        return DOC_YEARS_CACHE[doc_id]
    years = [int(m.group(1)) for m in YEAR_PATTERN.finditer(doc_text or "")]
    years = sorted(set(years))
    DOC_YEARS_CACHE[doc_id] = years
    return years

def compute_temporal_score(tc: Optional[TemporalConstraintV1], doc_years: List[int], max_span: int = 20) -> float:
    if tc is None: return 1.0
    if not doc_years: return 0.5

    def triangular(distance: float) -> float:
        return max(0.0, 1.0 - (distance / float(max_span)))

    if tc.type == "point" and tc.year is not None:
        diff = min(abs(y - tc.year) for y in doc_years)
        return triangular(diff)

    if tc.type == "range" and tc.start_year is not None:
        if any(tc.start_year <= y <= tc.end_year for y in doc_years):
            return 1.0
        distances = []
        for y in doc_years:
            if y < tc.start_year: distances.append(tc.start_year - y)
            elif y > tc.end_year: distances.append(y - tc.end_year)
        if not distances: return 0.5
        return triangular(min(distances))

    return 1.0

def mrag_rerank_1(
    question_text: str,
    candidate_passages: List[str],
    candidate_ids: List[int],
    model,
    tokenizer,
    base_scores: np.ndarray = None,
    blend_weight: float = 0.0,
    temporal_weight: float = 1.0,
    window_emb_tensor=None,
    doc_window_map=None,
    **kwargs
):
    if not candidate_passages:
        return [], []

    # 1. Decompose
    mc_text, tc = decompose_question_temporal_v1(question_text)

    # 2. Dense MaxSim (Fast)
    if window_emb_tensor is not None and doc_window_map is not None:
        granular_scores = get_dense_maxsim_scores_fast(
            mc_text, candidate_ids, model, tokenizer, window_emb_tensor, doc_window_map
        )
    else:
        # Fallback (Slow) - only used if pre-computation failed
        if _get_doc_id(candidate_ids[0]) not in PRETOKENIZED_WINDOWS:
            pretokenize_passages(candidate_passages, candidate_ids)
        granular_scores = [0.0] * len(candidate_ids) 

    granular_scores = np.array(granular_scores, dtype=np.float32)

    # 3. Normalize
    base_scores_norm = np.zeros(len(candidate_ids), dtype=np.float32)
    if base_scores is not None and len(base_scores) > 0:
        bs = np.array(base_scores[:len(candidate_ids)], dtype=np.float32)
        if bs.max() > bs.min():
            base_scores_norm = (bs - bs.min()) / (bs.max() - bs.min())
        else:
            base_scores_norm = bs

    if granular_scores.max() > granular_scores.min():
        granular_scores_norm = (granular_scores - granular_scores.min()) / (granular_scores.max() - granular_scores.min())
    else:
        granular_scores_norm = granular_scores

    # 4. Temporal Scoring
    final_scores = {}
    for i, cid in enumerate(candidate_ids):
        doc_years = get_doc_years(cid, candidate_passages[i])
        
        semantic_score = (blend_weight * float(base_scores_norm[i]) + (1.0 - blend_weight) * float(granular_scores_norm[i]))
        t_score = compute_temporal_score(tc, doc_years) if tc else 1.0
        
        hybrid_factor = (temporal_weight * t_score + (1.0 - temporal_weight))
        final_scores[cid] = semantic_score * hybrid_factor

    ranked_ids = sorted(final_scores, key=final_scores.get, reverse=True)
    return ranked_ids

def evaluate_mrag_on_caqa(
    index,
    model,
    tokenizer,
    retrieval_questions,
    gold_ids,
    passages_list,
    use_mrag: bool,
    desc="",
    top_k=100,
    full_questions=None,
    **kwargs
):
    print(f"[EVAL] {desc} | q={len(retrieval_questions)} | mrag={use_mrag}")
    
    q_embs = encode_texts(model, tokenizer, retrieval_questions)
    scores, ids = index.search(q_embs, top_k)
    
    metrics = {f"hit@{k}": 0.0 for k in EVAL_KS}
    metrics.update({f"mrr@{k}": 0.0 for k in EVAL_KS})
    
    # Using tqdm for progress
    for qi, gold in enumerate(tqdm(gold_ids, desc="Reranking")):
        cand_ids = [int(cid) for cid in ids[qi] if 0 <= cid < len(passages_list)]
        cand_scores = scores[qi][:len(cand_ids)]
        cand_texts = [passages_list[cid] for cid in cand_ids]
        
        if use_mrag:
            q_text = full_questions[qi] if full_questions else retrieval_questions[qi]
            ranked_ids = mrag_rerank_1(
                q_text, cand_texts, cand_ids, model, tokenizer,
                base_scores=cand_scores, blend_weight=0.0, temporal_weight=1.0,
                **kwargs
            )
        else:
            ranked_ids = cand_ids
            
        for k in EVAL_KS:
            if gold in ranked_ids[:k]:
                metrics[f"hit@{k}"] += 1.0
                for rank, pid in enumerate(ranked_ids[:k], start=1):
                    if pid == gold:
                        metrics[f"mrr@{k}"] += 1.0 / rank
                        break
                        
    total = len(gold_ids) if gold_ids else 1
    metrics = {k: v / total for k, v in metrics.items()}
    print(f"Results for {desc}: {metrics}")
    return metrics

def load_config():
    paths = ['config.yaml', '../config.yaml', 'src/config.yaml', 'configs/config.yaml']
    for p in paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("Could not find config.yaml")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=ENCODING_BATCH_SIZE)
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--output', type=str, default='outputs/mrag_eval_results_fast.json')
    args = parser.parse_args()

    config = load_config()
    base_name = config['models']['base_contriever']['name']
    time_path = config['models']['time_aware_contriever']['output_dir']

    print(f"Loading Base: {base_name}")
    print(f"Loading Time-Aware: {time_path}")
    
    base_tokenizer = AutoTokenizer.from_pretrained(base_name)
    base_model = AutoModel.from_pretrained(base_name).to(DEVICE).eval()
    
    # Use Base tokenizer to prevent config mismatch
    time_tokenizer = AutoTokenizer.from_pretrained(base_name)
    try:
        time_model = AutoModel.from_pretrained(time_path).to(DEVICE).eval()
    except Exception as e:
        print(f"WARNING: Failed to load time-aware model from {time_path}: {e}")
        time_model = base_model

    print("Loading ChroniclingAmericaQA...")
    caqa = load_dataset("Bhawna/ChroniclingAmericaQA", split="validation")

    passages = []
    passage_to_id = {}
    questions_full = []
    gold_ids_full = []

    # --- FIX IS HERE ---
    for ex in tqdm(caqa, desc="Processing Data"):
        q = ex.get('question') or ex.get('query')
        # Robustly checking for 'context', 'passage', or 'positive_passage'
        p = ex.get('context') or ex.get('positive_passage') or ex.get('passage')
        
        if not q or not p: continue
        
        if p not in passage_to_id:
            pid = len(passages)
            passage_to_id[p] = pid
            passages.append(p)
        else:
            pid = passage_to_id[p]
        questions_full.append(q)
        gold_ids_full.append(pid)
    # -------------------

    # Year Subset
    year_regex = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")
    questions_year = []
    gold_ids_year = []
    for q, gid in zip(questions_full, gold_ids_full):
        if year_regex.search(q):
            questions_year.append(q)
            gold_ids_year.append(gid)

    if args.sample_size:
        print(f"Subsampling to {args.sample_size}")
        questions_year_eval = questions_year[:args.sample_size]
        gold_ids_year_eval = gold_ids_year[:args.sample_size]
    else:
        questions_year_eval = questions_year
        gold_ids_year_eval = gold_ids_year

    print(f"Total Passages: {len(passages)}")
    print(f"Questions to Eval: {len(questions_year_eval)}")

    if len(passages) == 0:
        raise ValueError("No passages loaded! Check dataset keys.")

    # Build Indexes
    base_index = build_faiss_index(base_model, base_tokenizer, passages)
    time_index = build_faiss_index(time_model, time_tokenizer, passages)

    # --- PRE-COMPUTE STEP ---
    print("\n[INFO] Pre-computing window embeddings (Global Speedup)...")
    
    print(">>> Base Model Windows")
    base_win_tensor, base_win_map = precompute_window_embeddings(base_model, base_tokenizer, passages)
    
    print(">>> Time-Aware Model Windows")
    time_win_tensor, time_win_map = precompute_window_embeddings(time_model, time_tokenizer, passages)

    # --- EVALUATIONS ---
    results = {}
    q_mc_year = [extract_mc(q) for q in questions_year_eval]

    # 1. Base Only (No MRAG)
    results['base_only'] = evaluate_mrag_on_caqa(
        base_index, base_model, base_tokenizer,
        q_mc_year, gold_ids_year_eval, passages,
        use_mrag=False, desc="Base Only"
    )

    # 2. Time-Aware Only (No MRAG)
    results['time_aware_only'] = evaluate_mrag_on_caqa(
        time_index, time_model, time_tokenizer,
        questions_year_eval, gold_ids_year_eval, passages,
        use_mrag=False, desc="Time-Aware Only"
    )

    # 3. MRAG + Base
    results['mrag_base'] = evaluate_mrag_on_caqa(
        base_index, base_model, base_tokenizer,
        q_mc_year, gold_ids_year_eval, passages,
        use_mrag=True, full_questions=questions_year_eval,
        desc="MRAG + Base",
        window_emb_tensor=base_win_tensor,
        doc_window_map=base_win_map
    )

    # 4. MRAG + Time-Aware
    results['mrag_time_aware'] = evaluate_mrag_on_caqa(
        time_index, time_model, time_tokenizer,
        questions_year_eval, gold_ids_year_eval, passages,
        use_mrag=True, full_questions=questions_year_eval,
        desc="MRAG + Time-Aware",
        window_emb_tensor=time_win_tensor,
        doc_window_map=time_win_map
    )

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Results saved to {args.output}\n")

if __name__ == "__main__":
    main()