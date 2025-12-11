"""
ChroniclingQA Evaluation for Time-Aware RAG
Exactly matches the notebook implementation
"""

import os
import re
import json
import yaml
import torch
import logging
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
import numpy as np
from datasets import load_dataset
import faiss
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from notebook
CAQA_SPLIT = "validation"  # change to "test" or "train" if desired
YEAR_SUBSET_LIMIT = None  # evaluate all year-explicit questions
RETRIEVE_TOPK = 100
YEAR_REGEX = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")

RESULTS = []  # Global results tracking like notebook


# Helper functions - exact notebook implementation
def _pick_field(ex, field_names):
    """Pick the first available field from a list"""
    for field in field_names:
        if field in ex and ex[field] is not None:
            txt = str(ex[field]).strip()
            if txt:
                return txt
    return None

def _hit_at_k(ranked_ids, gold_id, k):
    return 1.0 if gold_id in ranked_ids[:k] else 0.0

def _mrr_at_k(ranked_ids, gold_id, k):
    for rank, pid in enumerate(ranked_ids[:k], start=1):
        if pid == gold_id:
            return 1.0 / rank
    return 0.0

def mean_pooling(last_hidden_state, attention_mask):
    """Mean pooling - exact notebook implementation"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_texts(model, tokenizer, texts, batch_size=64, max_len=256):
    """Encode texts with Contriever - exact notebook implementation"""
    device = next(model.parameters()).device
    model.eval()
    outs = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        tok = tokenizer(batch_texts, padding=True, truncation=True, 
                       max_length=max_len, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = model(**tok)
                embeddings = mean_pooling(outputs.last_hidden_state, tok['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            outs.append(embeddings.cpu().numpy().astype("float32"))
    
    return np.vstack(outs) if outs else np.zeros((0, model.config.hidden_size), "float32")

def build_faiss_index(model, tokenizer, passages):
    """Build FAISS index - exact notebook implementation"""
    embs = encode_texts(model, tokenizer, passages)
    dim = embs.shape[1]
    index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
    ids = np.arange(len(passages))
    index.add_with_ids(embs, ids)
    return index, dim

def retrieve_candidates_caqa(index, model, tokenizer, question_texts: List[str], top_k: int = RETRIEVE_TOPK):
    q_embs = encode_texts(model, tokenizer, question_texts)
    scores, ids = index.search(q_embs, top_k)
    return scores, ids

def evaluate_contriever_on_caqa(
    index,
    model,
    tokenizer,
    questions,
    gold_ids,
    desc="Unknown Model",
    k_list=(1, 5, 10, 20),
    question_preprocessor=None,
):
    """Evaluate Contriever on ChroniclingQA - exact notebook implementation"""
    if question_preprocessor is not None:
        eval_questions = [question_preprocessor(q) for q in questions]
    else:
        eval_questions = questions

    q_embs = encode_texts(model, tokenizer, eval_questions)
    scores, ids = index.search(q_embs, max(k_list))

    metrics = {f"hit@{k}": 0.0 for k in k_list}
    metrics.update({f"mrr@{k}": 0.0 for k in k_list})

    for qi, gold in enumerate(gold_ids):
        ranked = ids[qi]
        for k in k_list:
            metrics[f"hit@{k}"] += _hit_at_k(ranked, gold, k)
            metrics[f"mrr@{k}"] += _mrr_at_k(ranked, gold, k)

    total = float(len(gold_ids)) if gold_ids else 1.0
    metrics = {k: v / total for k, v in metrics.items()}
    print(f"{desc} Results:", metrics)
    
    RESULTS.append({
        "Model": desc,
        "Split": "full",
        **{k.upper(): v for k, v in metrics.items()}
    })
    return metrics


class ChroniclingQAEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initialized evaluator on {self.device}")
    
    def load_models(self):
        """Load models - exact notebook approach"""
        # Load base model
        base_model = AutoModel.from_pretrained("facebook/contriever-msmarco").to(self.device)
        base_tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
        base_model.eval()
        logger.info("Loaded base model: facebook/contriever-msmarco")
        
        # Load time-aware model
        time_model_path = "contriever_finetuned_NEW_20k"
        if os.path.exists(time_model_path):
            time_model = AutoModel.from_pretrained(time_model_path).to(self.device)
            time_tokenizer = AutoTokenizer.from_pretrained(time_model_path)
            time_model.eval()
            logger.info(f"Loaded time-aware model from {time_model_path}")
        else:
            logger.warning(f"Time-aware model not found at {time_model_path}")
            time_model, time_tokenizer = None, None
        
        return (base_model, base_tokenizer), (time_model, time_tokenizer)
    
    def load_caqa_data(self):
        """Load ChroniclingQA data - exact notebook implementation"""
        print("Loading ChroniclingAmericaQA split:", CAQA_SPLIT)
        caqa = load_dataset("Bhawna/ChroniclingAmericaQA", split=CAQA_SPLIT)
        print("Columns:", caqa.column_names)

        ca_passage_text_to_id = {}
        ca_passages_list: List[str] = []
        ca_passage_ids_list: List[int] = []
        ca_questions: List[str] = []
        ca_gold_ids: List[int] = []

        for ex in caqa:
            question = _pick_field(ex, ["question", "query", "input", "prompt", "text"])
            passage = _pick_field(ex, [
                "positive_passage", "passage", "context", "document", "evidence", "target", "passage_text", "ctx"
            ])
            if question is None or passage is None:
                continue
            if passage not in ca_passage_text_to_id:
                pid = len(ca_passages_list)
                ca_passage_text_to_id[passage] = pid
                ca_passages_list.append(passage)
                ca_passage_ids_list.append(pid)
            else:
                pid = ca_passage_text_to_id[passage]
            ca_questions.append(question)
            ca_gold_ids.append(pid)

        # Build year-explicit subset - exact notebook logic
        ca_questions_year = []
        ca_gold_ids_year = []
        for q, gid in zip(ca_questions, ca_gold_ids):
            if YEAR_REGEX.search(q):
                ca_questions_year.append(q)
                ca_gold_ids_year.append(gid)

        if YEAR_SUBSET_LIMIT is not None:
            ca_questions_year = ca_questions_year[:YEAR_SUBSET_LIMIT]
            ca_gold_ids_year = ca_gold_ids_year[:YEAR_SUBSET_LIMIT]

        print(f"Full CAQA: {len(ca_questions)} questions, {len(ca_passages_list)} passages")
        print(f"Year-explicit subset: {len(ca_questions_year)} questions")
        
        return {
            'passages_list': ca_passages_list,
            'questions_full': ca_questions,
            'gold_ids_full': ca_gold_ids,
            'questions_year': ca_questions_year,
            'gold_ids_year': ca_gold_ids_year
        }
    
    def load_t5_test_data(self):
        """Load T5-generated test questions from training data"""
        print("\nLoading T5-generated test set...")
        
        # Load FineWeb passages
        fineweb_path = "data/fineweb/fineweb_passages.json"
        if not os.path.exists(fineweb_path):
            print(f"Warning: {fineweb_path} not found, skipping T5 evaluation")
            return None
            
        with open(fineweb_path, 'r') as f:
            fineweb_data = json.load(f)
        
        # Load generated questions
        questions_path = "data/generated_questions/sample_generated_questions.json"
        if not os.path.exists(questions_path):
            print(f"Warning: {questions_path} not found, skipping T5 evaluation")
            return None
            
        with open(questions_path, 'r') as f:
            questions_data = json.load(f)
        
        # Create 80/20 split like in training (using same random seed for consistency)
        import random
        random.seed(42)
        
        # Create question-passage pairs
        all_pairs = []
        passages_dict = {p['id']: p['text'] for p in fineweb_data}  # Use numeric IDs
        
        # Try to match questions with passages
        for q_item in questions_data:
            passage_id_str = q_item['passage_id']  # e.g., "sample_335243"
            
            # Extract numeric ID from "sample_XXXXX" format
            if passage_id_str.startswith('sample_'):
                numeric_id = int(passage_id_str.replace('sample_', ''))
                if numeric_id in passages_dict:
                    all_pairs.append({
                        'question': q_item['question'], 
                        'passage': passages_dict[numeric_id],
                        'passage_id': numeric_id
                    })
        
        print(f"Found {len(all_pairs)} matching question-passage pairs")
        
        # If no matches found, return None to skip T5 evaluation
        if len(all_pairs) == 0:
            print("No matching passage IDs found between questions and passages. Skipping T5 evaluation.")
            return None
        
        # Create train/test split (80/20)
        random.shuffle(all_pairs)
        split_idx = int(0.8 * len(all_pairs))
        test_pairs = all_pairs[split_idx:]  # Use test split
        
        # Build test data structures
        t5_passage_text_to_id = {}
        t5_passages_list = []
        t5_questions = []
        t5_gold_ids = []
        
        for pair in test_pairs:
            question = pair['question']
            passage = pair['passage']
            
            if passage not in t5_passage_text_to_id:
                pid = len(t5_passages_list)
                t5_passage_text_to_id[passage] = pid
                t5_passages_list.append(passage)
            else:
                pid = t5_passage_text_to_id[passage]
                
            t5_questions.append(question)
            t5_gold_ids.append(pid)
        
        # Build year-explicit subset for T5 data
        t5_questions_year = []
        t5_gold_ids_year = []
        for q, gid in zip(t5_questions, t5_gold_ids):
            if YEAR_REGEX.search(q):
                t5_questions_year.append(q)
                t5_gold_ids_year.append(gid)
        
        print(f"T5 Test Set: {len(t5_questions)} questions, {len(t5_passages_list)} passages")
        print(f"T5 Year-explicit subset: {len(t5_questions_year)} questions")
        
        return {
            'passages_list': t5_passages_list,
            'questions_full': t5_questions,
            'gold_ids_full': t5_gold_ids,
            'questions_year': t5_questions_year,
            'gold_ids_year': t5_gold_ids_year
        }
    
    def run_evaluation(self):
        """Run complete evaluation - exact notebook implementation"""
        # Load models
        (base_model, base_tokenizer), (time_model, time_tokenizer) = self.load_models()
        
        # T5-Generated Questions Evaluation (IN-DOMAIN)
        print("="*80)
        print("T5-GENERATED QUESTIONS EVALUATION (IN-DOMAIN)")
        print("="*80)
        t5_data = self.load_t5_test_data()
        
        if t5_data is not None:
            # Build T5 FAISS indices
            logger.info("Building T5 FAISS indices...")
            t5_base_index, emb_dim = build_faiss_index(base_model, base_tokenizer, t5_data['passages_list'])
            logger.info(f"T5 baseline index built with {t5_base_index.ntotal} vectors (dim={emb_dim})")
            
            if time_model is not None:
                t5_time_index, _ = build_faiss_index(time_model, time_tokenizer, t5_data['passages_list'])
                logger.info(f"T5 time-aware index built with {t5_time_index.ntotal} vectors")
            else:
                t5_time_index = t5_base_index
                time_model, time_tokenizer = base_model, base_tokenizer
                logger.warning("Using base model for time-aware evaluation")
            
            # T5 Full dataset evaluation
            logger.info("Running T5 evaluations...")
            
            metrics_t5_base_full = evaluate_contriever_on_caqa(
                t5_base_index,
                base_model,
                base_tokenizer,
                t5_data['questions_full'],
                t5_data['gold_ids_full'],
                desc="T5 In-Domain Full [BASE]",
                question_preprocessor=None,
            )
            
            metrics_t5_time_full = evaluate_contriever_on_caqa(
                t5_time_index,
                time_model,
                time_tokenizer,
                t5_data['questions_full'],
                t5_data['gold_ids_full'],
                desc="T5 In-Domain Full [TIMEAWARE]",
                question_preprocessor=None,
            )
            
            # T5 Year subset evaluation (if available)
            if t5_data['questions_year']:
                metrics_t5_base_year = evaluate_contriever_on_caqa(
                    t5_base_index,
                    base_model,
                    base_tokenizer,
                    t5_data['questions_year'],
                    t5_data['gold_ids_year'],
                    desc="T5 In-Domain Year-Subset [BASE]",
                    question_preprocessor=None,
                )
                
                metrics_t5_time_year = evaluate_contriever_on_caqa(
                    t5_time_index,
                    time_model,
                    time_tokenizer,
                    t5_data['questions_year'],
                    t5_data['gold_ids_year'],
                    desc="T5 In-Domain Year-Subset [TIMEAWARE]",
                    question_preprocessor=None,
                )
            else:
                print("No year-explicit questions found in T5 dataset")
        
        # ChroniclingQA Evaluation (OUT-OF-DOMAIN)
        print("\n" + "="*80)
        print("CHRONICLINGQA EVALUATION (OUT-OF-DOMAIN)")
        print("="*80)
        data = self.load_caqa_data()
        
        # Build FAISS indices - exact notebook approach
        logger.info("Building ChroniclingQA FAISS indices...")
        caq_base_index, emb_dim = build_faiss_index(base_model, base_tokenizer, data['passages_list'])
        logger.info(f"ChroniclingQA baseline index built with {caq_base_index.ntotal} vectors (dim={emb_dim})")
        
        if time_model is not None:
            caq_time_index, _ = build_faiss_index(time_model, time_tokenizer, data['passages_list'])
            logger.info(f"ChroniclingQA time-aware index built with {caq_time_index.ntotal} vectors")
        else:
            caq_time_index = caq_base_index
            time_model, time_tokenizer = base_model, base_tokenizer
            logger.warning("Using base model for time-aware evaluation")
        
        # ChroniclingQA Evaluation
        logger.info("Running ChroniclingQA evaluations...")
        
        # Full dataset evaluation
        metrics_caqa_base_full = evaluate_contriever_on_caqa(
            caq_base_index,
            base_model,
            base_tokenizer,
            data['questions_full'],
            data['gold_ids_full'],
            desc="CAQA Out-of-Domain Full [BASE]",
            question_preprocessor=None,
        )
        
        metrics_caqa_time_full = evaluate_contriever_on_caqa(
            caq_time_index,
            time_model,
            time_tokenizer,
            data['questions_full'],
            data['gold_ids_full'],
            desc="CAQA Out-of-Domain Full [TIMEAWARE]",
            question_preprocessor=None,
        )
        
        # Year subset evaluation (if available)
        if data['questions_year']:
            metrics_caqa_base_year = evaluate_contriever_on_caqa(
                caq_base_index,
                base_model,
                base_tokenizer,
                data['questions_year'],
                data['gold_ids_year'],
                desc="CAQA Out-of-Domain Year-Subset [BASE]",
                question_preprocessor=None,
            )
            
            metrics_caqa_time_year = evaluate_contriever_on_caqa(
                caq_time_index,
                time_model,
                time_tokenizer,
                data['questions_year'],
                data['gold_ids_year'],
                desc="CAQA Out-of-Domain Year-Subset [TIMEAWARE]",
                question_preprocessor=None,
            )
        else:
            logger.warning("No year-explicit questions found in dataset")
        
        return RESULTS
    
    def save_results(self, results, output_dir="outputs"):
        """Save evaluation results - notebook style"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame from results
        df = pd.DataFrame(results)
        
        # Save as CSV 
        csv_path = os.path.join(output_dir, 'chroniclingqa_results.csv')
        df.to_csv(csv_path, index=False)
        
        # Convert list format to dictionary format for plotting script compatibility
        results_dict = {}
        for result in results:
            model_name = result['Model'].replace(' ', '_').lower()
            results_dict[model_name] = {
                'hit@1': result.get('HIT@1', 0),
                'hit@5': result.get('HIT@5', 0), 
                'hit@10': result.get('HIT@10', 0),
                'hit@20': result.get('HIT@20', 0),
                'mrr@1': result.get('MRR@1', 0),
                'mrr@5': result.get('MRR@5', 0),
                'mrr@10': result.get('MRR@10', 0),
                'mrr@20': result.get('MRR@20', 0),
                'recall@1': result.get('HIT@1', 0),  # Recall is same as hit for plotting
                'recall@5': result.get('HIT@5', 0),
                'recall@10': result.get('HIT@10', 0)
            }
        
        # Save as JSON (dictionary format for plotting script)
        json_path = os.path.join(output_dir, 'chroniclingqa_results.json')
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
        
        # Print summary - exact notebook style
        print("\n" + "="*80)
        print("CHRONICLINGQA EVALUATION RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)


def main():
    # Initialize evaluator (no config needed)
    evaluator = ChroniclingQAEvaluator()
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    # Save results
    evaluator.save_results(results)
    
    logger.info("✓ ChroniclingQA evaluation completed successfully!")


if __name__ == "__main__":
    main()