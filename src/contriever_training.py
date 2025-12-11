"""
Contriever Fine-tuning for Time-Aware RAG
Exactly matches the notebook implementation - simple triplet training with MarginRankingLoss
"""

import os
import gc
import re
import json
import yaml
import torch
import random
import shutil
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mirror notebook constants exactly
YEAR_REGEX = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")

# Constants from notebook
CONTRIEVER_BASE = "facebook/contriever-msmarco"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16


class ContrieverTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyperparams - exactly match notebook  
        self.baseline_model = CONTRIEVER_BASE
        self.t5_qg_model = "valhalla/t5-base-qg-hl"
        self.fineweb_sample_size = config['data']['fineweb']['sample_size']
        self.max_passage_chars = config['data']['fineweb']['max_passage_chars']
        self.num_qg_passages = 15000  # notebook constant
        self.qg_batch_size = 64
        self.mining_pool_k = 100
        self.max_positives = 3
        self.max_negatives = 6
        self.semantic_threshold = 0.7
        self.train_batch_size = 64
        
        # Training hyperparams (exactly from notebook)
        self.train_epochs_hybrid = 14
        self.micro_batch_size = 32
        self.grad_acc_steps = 8
        self.train_lr_hybrid = 1e-5
        self.triplet_margin = 1.0
        self.max_len = 256
        self.amp_dtype = torch.float16
        
        self.ft_out_dir = "contriever_finetuned_NEW_20k"
        
        logger.info(f"Initialized trainer on {self.device}")
        
    def _norm(self, s: str) -> str:
        """Text normalization - exact notebook implementation"""
        s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
        return re.sub(r"\s+", " ", s).strip()
    
    def get_years_from_text(self, text: str) -> set:
        """Extract years from text - exact notebook implementation"""
        return set(YEAR_REGEX.findall(text))
    
    def mean_pooling(self, last_hidden_state, attention_mask):
        """Mean pooling - exact notebook implementation"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    @torch.no_grad()
    def encode_contriever(self, model, tokenizer, texts, max_len=256, batch=64):
        """Encode texts with Contriever - exact notebook implementation"""
        model.eval()
        outs = []
        for i in tqdm(range(0, len(texts), batch), desc="Encoding"):
            batch_texts = texts[i:i + batch]
            tok = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
            tok = {k: v.to(self.device) for k, v in tok.items()}
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(**tok)
                embeddings = self.mean_pooling(outputs.last_hidden_state, tok['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            outs.append(embeddings.cpu().numpy().astype("float32"))
        return np.vstack(outs) if outs else np.zeros((0, model.config.hidden_size), "float32")
    
    def build_faiss_index(self, model, tokenizer, passages_list, passage_ids_list, out_dir, index_path):
        """Build FAISS index - exact notebook implementation"""
        print(f"Building FAISS index in {out_dir}...")
        dim = model.config.hidden_size
        index_flat = faiss.IndexFlatIP(dim)
        
        # Convert string IDs to integers for FAISS (FAISS requires int64 IDs)
        id_mapping = {}
        int_ids = []
        for i, pid in enumerate(passage_ids_list):
            int_id = i  # Use sequential integers
            id_mapping[int_id] = pid  # Map back to original ID
            int_ids.append(int_id)
        
        ids = np.array(int_ids, dtype=np.int64)
        embs = self.encode_contriever(model, tokenizer, passages_list, batch=self.train_batch_size * 2)
        index_idmap = faiss.IndexIDMap2(index_flat)
        index_idmap.add_with_ids(embs, ids)
        faiss.write_index(index_idmap, index_path)
        print(f"Built FLAT index: {index_idmap.ntotal:,} vectors")
        
        # Store the ID mapping for later use
        self.id_mapping = id_mapping
        self.reverse_id_mapping = {pid: int_id for int_id, pid in id_mapping.items()}
        
        return index_idmap
    
    def load_fineweb_data(self):
        """Load FineWeb data - matches notebook approach"""
        fineweb_path = os.path.join(self.config['data']['fineweb']['output_path'], 'fineweb_passages.json')
        
        if not os.path.exists(fineweb_path):
            logger.error(f"FineWeb data not found at {fineweb_path}")
            logger.error("Please run fineweb_loader.py first")
            return []
        
        with open(fineweb_path, 'r') as f:
            data = json.load(f)
        
        # Convert to notebook format (id, text, title)
        train_passages_all = [(item['id'], item['text'], item['title']) for item in data]
        logger.info(f"Loaded {len(train_passages_all)} FineWeb passages")
        
        return train_passages_all
    
    @torch.no_grad()
    def generate_temporal_questions_batch(self, qg_model, qg_tok, passages, years, max_new_tokens=64):
        """Generate temporal questions - exact notebook implementation"""
        prompts = [f"generate question about {y}: {p}" for p, y in zip(passages, years)]
        
        inputs = qg_tok(prompts, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(qg_model.device) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=self.amp_dtype, enabled=torch.cuda.is_available()):
            outputs = qg_model.generate(**inputs, max_length=max_new_tokens, num_beams=4, early_stopping=True)
        return qg_tok.batch_decode(outputs, skip_special_tokens=True)
    
    def generate_synthetic_pairs(self, train_passages_all):
        """Generate synthetic temporal question-passage pairs - exact notebook implementation"""
        print(f"Loading T5 model: {self.t5_qg_model}...")
        qg_tokenizer = T5Tokenizer.from_pretrained(self.t5_qg_model)
        qg_model = T5ForConditionalGeneration.from_pretrained(self.t5_qg_model).to(self.device)
        qg_model.eval()
        
        if len(train_passages_all) > self.num_qg_passages:
            print(f"Sampling {self.num_qg_passages} passages for QG...")
            passages_to_gen = random.sample(train_passages_all, self.num_qg_passages)
        else:
            passages_to_gen = train_passages_all
        
        synthetic_pairs = []  # (question, passage_text, passage_id)
        passage_batch, passage_info, year_batch = [], [], []
        
        print(f"Generating {len(passages_to_gen)} synthetic TEMPORAL questions...")
        for (pid, text, title) in tqdm(passages_to_gen):
            years = self.get_years_from_text(text)
            if not years:
                continue
            first_year = sorted(list(years))[0]
            passage_batch.append(text)
            year_batch.append(first_year)
            passage_info.append((pid, text))
            
            if len(passage_batch) >= self.qg_batch_size:
                generated_questions = self.generate_temporal_questions_batch(qg_model, qg_tokenizer, passage_batch, year_batch)
                for i, q in enumerate(generated_questions):
                    if q:
                        p_id, p_text = passage_info[i]
                        synthetic_pairs.append((q, p_text, p_id))
                passage_batch, passage_info, year_batch = [], [], []
        
        # Process remaining batch
        if passage_batch:
            generated_questions = self.generate_temporal_questions_batch(qg_model, qg_tokenizer, passage_batch, year_batch)
            for i, q in enumerate(generated_questions):
                if q:
                    p_id, p_text = passage_info[i]
                    synthetic_pairs.append((q, p_text, p_id))
        
        print(f"Created {len(synthetic_pairs)} synthetic TEMPORAL (question, positive_passage) pairs.")
        
        # Cleanup
        del qg_model, qg_tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return synthetic_pairs
    
    def create_train_test_split(self, synthetic_pairs):
        """Create 80/20 split - exact notebook implementation"""
        print("Creating 80/20 Train/Test Split...")
        train_set, test_set = train_test_split(synthetic_pairs, test_size=0.2, random_state=42)
        print(f"Temporal Training set size: {len(train_set)}")
        print(f"Temporal Test set size: {len(test_set)}")
        
        corpus_passages_map = {pid: text for (q, text, pid) in synthetic_pairs}
        corpus_passages_list = list(corpus_passages_map.values())
        corpus_passage_ids_list = list(corpus_passages_map.keys())
        print(f"Total passages in our T5 dataset: {len(corpus_passages_map)}")
        
        return train_set, test_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list
    
    def mine_hard_negatives(self, train_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list):
        """Mine temporal hard negatives - exact notebook implementation"""
        print("Loading BASELINE Contriever model for mining...")
        contriever_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        contriever_model = AutoModel.from_pretrained(self.baseline_model).to(self.device)
        contriever_model.eval()
        
        print(f"Building FAISS index for {len(corpus_passages_map)} passages...")
        mining_dir = "contriever_mining_index_fineweb_20k"
        mining_index_path = os.path.join(mining_dir, "mining.index")
        shutil.rmtree(mining_dir, ignore_errors=True)
        os.makedirs(mining_dir, exist_ok=True)
        
        index_mining = self.build_faiss_index(
            contriever_model, contriever_tokenizer,
            corpus_passages_list, corpus_passage_ids_list,
            mining_dir, mining_index_path
        )
        
        print("Mining for augmented (1-to-N) temporal hard negatives...")
        triplet_examples = []
        questions_to_mine = [ex[0] for ex in train_set]
        q_embs = self.encode_contriever(contriever_model, contriever_tokenizer, questions_to_mine)
        search_results_D, search_results_I = index_mining.search(q_embs, self.mining_pool_k)
        
        for i in tqdm(range(len(train_set)), desc="Finding negatives"):
            q, p_pos_text, p_pos_id = train_set[i]
            pos_years = self.get_years_from_text(p_pos_text)
            if not pos_years:
                continue
            
            scores, passage_ids = search_results_D[i], search_results_I[i]
            other_positives, hard_negatives = [p_pos_text], []
            
            # Get the integer ID for the positive passage
            p_pos_int_id = self.reverse_id_mapping.get(p_pos_id, -1)
            
            for score, int_pid in zip(scores, passage_ids):
                if int_pid == -1 or score < self.semantic_threshold:
                    break
                if int_pid == p_pos_int_id:
                    continue
                
                # Convert back to original string ID
                original_pid = self.id_mapping.get(int_pid)
                if not original_pid:
                    continue
                    
                p_cand_text = corpus_passages_map.get(original_pid)
                if not p_cand_text:
                    continue
                
                cand_years = self.get_years_from_text(p_cand_text)
                if not cand_years:
                    continue
                
                if pos_years == cand_years and len(other_positives) < self.max_positives:
                    other_positives.append(p_cand_text)
                elif pos_years != cand_years:
                    hard_negatives.append(p_cand_text)
            
            if not hard_negatives:
                continue
            
            for p_pos in other_positives:
                for p_neg in hard_negatives[:self.max_negatives]:
                    triplet_examples.append((q, p_pos, p_neg))
        
        print(f"Created {len(triplet_examples)} augmented triplet training examples.")
        
        # Cleanup
        del contriever_model, index_mining
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return triplet_examples
    
    def load_msmarco_triplets(self):
        """Load MS MARCO triplets - exact notebook implementation"""
        print("Loading MS MARCO (General Domain) Triplets...")
        print("Streaming 'sentence-transformers/msmarco-msmarco-distilbert-base-tas-b' (Config: triplet-hard)...")
        
        msmarco_stream = load_dataset(
            "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
            "triplet-hard",
            split="train",
            streaming=True,
        )
        
        msmarco_triplets = []
        target_count = 1000  # match original balance intent
        pbar = tqdm(total=target_count, desc="Collecting MS MARCO")
        
        for row in msmarco_stream:
            if len(msmarco_triplets) >= target_count:
                break
            
            q = row.get('query')
            p = row.get('positive')
            n = row.get('negative')
            
            if isinstance(n, list):
                n = random.choice(n)
            
            if q and p and n:
                msmarco_triplets.append((q, p, n))
                pbar.update(1)
        
        pbar.close()
        print(f"Collected {len(msmarco_triplets)} MS MARCO Triplets.")
        
        return msmarco_triplets
    
    def train_model(self, combined_triplets):
        """Train the model - exact notebook implementation"""
        print(f"Training Model (Hybrid 50/50) - {len(combined_triplets)} triplets...")
        
        # Triplet dataset exactly like notebook
        class TripletDataset(torch.utils.data.Dataset):
            def __init__(self, examples):
                self.examples = examples
            def __len__(self):
                return len(self.examples)
            def __getitem__(self, idx):
                return self.examples[idx]
        
        def collate_triplets(batch):
            questions = [ex[0] for ex in batch]
            texts_pos = [ex[1] for ex in batch]
            texts_neg = [ex[2] for ex in batch]
            q_inputs = contriever_tokenizer(questions, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt")
            p_pos_inputs = contriever_tokenizer(texts_pos, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt")
            p_neg_inputs = contriever_tokenizer(texts_neg, padding="longest", truncation=True, max_length=self.max_len, return_tensors="pt")
            return {"q_inputs": q_inputs, "p_pos_inputs": p_pos_inputs, "p_neg_inputs": p_neg_inputs}
        
        print("Loading Fresh Model for Training...")
        # Enable grads for training
        torch.set_grad_enabled(True)
        
        contriever_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        contriever_model_train = AutoModel.from_pretrained(self.baseline_model).to(self.device)
        contriever_model_train.train()
        contriever_model_train.gradient_checkpointing_enable()
        
        train_dataset = TripletDataset(combined_triplets)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            collate_fn=collate_triplets,
            num_workers=0,  # Changed from 1 to 0 to avoid multiprocessing issues
            pin_memory=False,  # Disabled for debugging
        )
        
        # Validate dataloader
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Dataloader batches: {len(train_dataloader)}")
        print(f"Expected batches per epoch: {len(combined_triplets) // self.micro_batch_size}")
        
        # Test first batch
        try:
            first_batch = next(iter(train_dataloader))
            print(f"First batch shape - Q: {first_batch['q_inputs']['input_ids'].shape}")
            print("Dataloader validation successful")
        except Exception as e:
            print(f"Dataloader error: {e}")
            return
        
        params = contriever_model_train.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.train_lr_hybrid)
        num_train_steps = len(train_dataloader) // self.grad_acc_steps * self.train_epochs_hybrid
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_train_steps)
        scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        triplet_loss_fct = torch.nn.MarginRankingLoss(margin=self.triplet_margin, reduction='mean')
        
        print(f"Starting Training: {len(combined_triplets)} triplets, {self.train_epochs_hybrid} epochs")
        print(f"Dataloader length: {len(train_dataloader)} batches per epoch")
        print(f"Expected steps per epoch (with grad acc): {len(train_dataloader) // self.grad_acc_steps}")
        
        for epoch in range(self.train_epochs_hybrid):
            total_loss = 0
            step_count = 0
            optimizer.zero_grad(set_to_none=True)
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.train_epochs_hybrid}")
            
            for step, batch in enumerate(pbar):
                q_inputs = {k: v.to(self.device) for k, v in batch["q_inputs"].items()}
                p_pos_inputs = {k: v.to(self.device) for k, v in batch["p_pos_inputs"].items()}
                p_neg_inputs = {k: v.to(self.device) for k, v in batch["p_neg_inputs"].items()}
                
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=torch.cuda.is_available()):
                    q_vectors = contriever_model_train(**q_inputs).last_hidden_state
                    p_pos_vectors = contriever_model_train(**p_pos_inputs).last_hidden_state
                    p_neg_vectors = contriever_model_train(**p_neg_inputs).last_hidden_state
                    
                    def quick_pool(last_hidden, mask):
                        mask_exp = mask.unsqueeze(-1).expand(last_hidden.size()).float()
                        return torch.sum(last_hidden * mask_exp, 1) / torch.clamp(mask_exp.sum(1), min=1e-9)
                    
                    q_emb = quick_pool(q_vectors, q_inputs['attention_mask'])
                    p_pos_emb = quick_pool(p_pos_vectors, p_pos_inputs['attention_mask'])
                    p_neg_emb = quick_pool(p_neg_vectors, p_neg_inputs['attention_mask'])
                    
                    pos_scores = (q_emb * p_pos_emb).sum(1)
                    neg_scores = (q_emb * p_neg_emb).sum(1)
                    
                    loss = triplet_loss_fct(pos_scores, neg_scores, torch.ones(q_emb.size(0)).to(self.device)) / self.grad_acc_steps
                
                scaler.scale(loss).backward()
                total_loss += loss.item() * self.grad_acc_steps
                
                if (step + 1) % self.grad_acc_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_count += 1
                
                pbar.set_postfix({
                    "Loss": loss.item() * self.grad_acc_steps,
                    "Step": step_count,
                    "Batch": step + 1
                })
            
            print(f"Epoch {epoch+1} Mean Loss: {total_loss / len(train_dataloader):.4f}")
            print(f"Completed {step_count} gradient steps in epoch {epoch+1}")
        
        # Save model - exact notebook approach
        print("Saving Model...")
        os.makedirs(self.ft_out_dir, exist_ok=True)
        contriever_model_train.save_pretrained(self.ft_out_dir)
        contriever_tokenizer.save_pretrained(self.ft_out_dir)
        print(f"Saved to {self.ft_out_dir}")
        
        # Optional: zip for download
        if os.path.exists(self.ft_out_dir):
            shutil.make_archive(self.ft_out_dir, 'zip', self.ft_out_dir)
            print(f"Packaged fine-tuned model to {self.ft_out_dir}.zip")
        
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Disable grads again
        torch.set_grad_enabled(False)
        
        logger.info("Training completed successfully!")
    
    def run_complete_training(self):
        """Run the complete training pipeline - matches notebook exactly"""
        logger.info("Starting complete Contriever training pipeline...")
        
        # Check if questions already exist (to avoid regenerating)
        questions_path = os.path.join(
            self.config['data']['generated_questions']['output_path'],
            'sample_generated_questions.json'
        )
        
        if os.path.exists(questions_path):
            logger.info("Found existing generated questions, skipping question generation...")
            # Load existing questions for training
            with open(questions_path, 'r') as f:
                existing_data = json.load(f)
            # Convert to synthetic pairs format: (question, passage, passage_id)  
            synthetic_pairs = []
            for item in existing_data:  # Use ALL generated questions like notebook
                synthetic_pairs.append((item['question'], item['passage'], item['passage_id']))
            logger.info(f"Loaded {len(synthetic_pairs)} existing question-passage pairs")
        else:
            logger.info("No existing questions found, generating new ones...")
            # Load FineWeb data
            train_passages_all = self.load_fineweb_data()
            if not train_passages_all:
                return False
            
            # Generate synthetic temporal pairs
            synthetic_pairs = self.generate_synthetic_pairs(train_passages_all)
            if not synthetic_pairs:
                logger.error("No synthetic pairs generated!")
                return False
        
        # Create train/test split
        train_set, test_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list = self.create_train_test_split(synthetic_pairs)
        
        # Mine hard negatives
        temporal_triplets = self.mine_hard_negatives(train_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list)
        
        # Load MS MARCO triplets
        msmarco_triplets = self.load_msmarco_triplets()
        
        # Combine and shuffle
        combined_triplets = temporal_triplets + msmarco_triplets
        random.shuffle(combined_triplets)
        print(f"Final Hybrid Training Set Size: {len(combined_triplets)} Triplets")
        
        # Clean up memory
        del msmarco_triplets
        gc.collect()
        
        # Train model
        self.train_model(combined_triplets)
        
        return True


def main():
    """Main training function"""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = ContrieverTrainer(config)
    
    # Run complete training pipeline
    success = trainer.run_complete_training()
    
    if success:
        logger.info("Contriever training completed successfully!")
    else:
        logger.error("Training failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())