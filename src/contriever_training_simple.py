"""
Simplified Contriever Training - Skip Question Generation
Uses existing questions to avoid memory issues
"""

import os
import json
import yaml
import torch
import random
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONTRIEVER_BASE = "facebook/contriever-msmarco"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.float16


class SimpleContrieverTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training hyperparams - exactly match notebook  
        self.baseline_model = CONTRIEVER_BASE
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
            with torch.autocast(self.device, dtype=self.amp_dtype, enabled=torch.cuda.is_available()):
                outputs = model(**tok)
                embeddings = self.mean_pooling(outputs.last_hidden_state, tok['attention_mask'])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            outs.append(embeddings.cpu().numpy().astype("float32"))
        return np.vstack(outs) if outs else np.zeros((0, model.config.hidden_size), "float32")
    
    def load_existing_questions(self):
        """Load existing questions to avoid regeneration"""
        questions_path = os.path.join(
            self.config['data']['generated_questions']['output_path'],
            'sample_generated_questions.json'
        )
        
        if not os.path.exists(questions_path):
            raise FileNotFoundError(f"Questions not found at {questions_path}")
        
        logger.info(f"Loading existing questions from {questions_path}")
        with open(questions_path, 'r') as f:
            existing_data = json.load(f)
        
        # Convert to synthetic pairs format: (question, passage, passage_id)  
        synthetic_pairs = []
        for item in existing_data:
            synthetic_pairs.append((item['question'], item['passage'], item['passage_id']))
        
        logger.info(f"Loaded {len(synthetic_pairs)} existing question-passage pairs")
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
    
    def build_faiss_index(self, model, tokenizer, passages_list, passage_ids_list, out_dir, index_path):
        """Build FAISS index - exact notebook implementation"""
        print(f"Building FAISS index in {out_dir}...")
        dim = model.config.hidden_size
        index_flat = faiss.IndexFlatIP(dim)
        ids = np.array(passage_ids_list, dtype=np.int64)
        embs = self.encode_contriever(model, tokenizer, passages_list, batch=self.train_batch_size * 2)
        index_idmap = faiss.IndexIDMap2(index_flat)
        index_idmap.add_with_ids(embs, ids)
        faiss.write_index(index_idmap, index_path)
        print(f"Built FLAT index: {index_idmap.ntotal:,} vectors")
        return index_idmap
    
    def mine_hard_negatives(self, train_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list):
        """Mine temporal hard negatives - simplified"""
        print("Loading BASELINE Contriever model for mining...")
        contriever_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        contriever_model = AutoModel.from_pretrained(self.baseline_model).to(self.device)
        contriever_model.eval()
        
        # Create some simple triplets from the data
        triplet_examples = []
        print("Creating simple triplets from existing data...")
        
        for i in range(min(1000, len(train_set))):  # Limit for faster training
            q, p_pos_text, p_pos_id = train_set[i]
            
            # Simple negative sampling - random passages
            negative_candidates = [text for pid, text in corpus_passages_map.items() if pid != p_pos_id]
            if negative_candidates:
                p_neg_text = random.choice(negative_candidates[:100])  # Limit choices
                triplet_examples.append((q, p_pos_text, p_neg_text))
        
        print(f"Created {len(triplet_examples)} simple triplet training examples.")
        
        # Cleanup
        del contriever_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return triplet_examples
    
    def train_model(self, triplet_examples):
        """Train the model - simplified version"""
        print(f"Training Model - {len(triplet_examples)} triplets...")
        
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
        torch.set_grad_enabled(True)
        
        contriever_tokenizer = AutoTokenizer.from_pretrained(self.baseline_model)
        contriever_model_train = AutoModel.from_pretrained(self.baseline_model).to(self.device)
        contriever_model_train.train()
        
        train_dataset = TripletDataset(triplet_examples)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.micro_batch_size,
            shuffle=True,
            collate_fn=collate_triplets,
            num_workers=0,
            pin_memory=False,
        )
        
        # Validate dataloader
        print(f"Dataset size: {len(train_dataset)}")
        print(f"Dataloader batches: {len(train_dataloader)}")
        
        params = contriever_model_train.parameters()
        optimizer = torch.optim.AdamW(params, lr=self.train_lr_hybrid)
        num_train_steps = len(train_dataloader) // self.grad_acc_steps * self.train_epochs_hybrid
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_train_steps)
        scaler = GradScaler(enabled=(self.device.type == 'cuda'))
        triplet_loss_fct = torch.nn.MarginRankingLoss(margin=self.triplet_margin, reduction='mean')
        
        print(f"Starting Training: {len(triplet_examples)} triplets, {self.train_epochs_hybrid} epochs")
        
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
        
        # Save model
        print("Saving Model...")
        os.makedirs(self.ft_out_dir, exist_ok=True)
        contriever_model_train.save_pretrained(self.ft_out_dir)
        contriever_tokenizer.save_pretrained(self.ft_out_dir)
        print(f"Saved to {self.ft_out_dir}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        torch.set_grad_enabled(False)
        logger.info("Training completed successfully!")
    
    def run_training(self):
        """Run simplified training pipeline"""
        logger.info("Starting simplified Contriever training...")
        
        # Load existing questions
        synthetic_pairs = self.load_existing_questions()
        
        # Create train/test split
        train_set, test_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list = self.create_train_test_split(synthetic_pairs)
        
        # Mine hard negatives (simplified)
        triplet_examples = self.mine_hard_negatives(train_set, corpus_passages_map, corpus_passages_list, corpus_passage_ids_list)
        
        # Train model
        self.train_model(triplet_examples)
        
        return True


def main():
    """Main training function"""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = SimpleContrieverTrainer(config)
    
    # Run training
    success = trainer.run_training()
    
    if success:
        logger.info("Contriever training completed successfully!")
    else:
        logger.error("Training failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())