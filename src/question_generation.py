"""
T5-based Question Generation for Time-Aware RAG
Generates questions from passages with temporal awareness
"""

import os
import json
import yaml
import torch
import logging
from typing import List, Dict, Any
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class T5QuestionGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load T5 model and tokenizer
        model_name = config['models']['t5_generator']['name']
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded T5 model: {model_name} on {self.device}")

    def generate_temporal_questions_batch(self, passages: List[str], years: List[str], max_new_tokens: int = 64) -> List[str]:
        """Generate temporal questions using T5 - matches notebook implementation exactly"""
        
        # Use the exact same prompt format as in the notebook
        prompts = [f"generate question about {y}: {p}" for p, y in zip(passages, years)]
        
        # Batch tokenization exactly like the notebook
        inputs = self.tokenizer(prompts, padding="longest", truncation=True, max_length=512, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate questions with exact same parameters as notebook
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                outputs = self.model.generate(**inputs, max_length=max_new_tokens, num_beams=4, early_stopping=True)
        
        # Decode exactly like the notebook
        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Clean up the outputs - remove the input prompts
        questions = []
        for i, decoded_text in enumerate(decoded):
            # Remove the input prompt from the output
            prompt = prompts[i]
            if decoded_text.startswith(prompt):
                question = decoded_text[len(prompt):].strip()
            else:
                question = decoded_text.strip()
            
            # Ensure it ends with a question mark
            if question and not question.endswith('?'):
                question += '?'
            
            questions.append(question)
        
        return questions

    def get_years_from_text(self, text: str):
        """Extract years from text - matches notebook implementation"""
        import re
        # Use the same regex as in the notebook
        YEAR_REGEX = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")
        return set(YEAR_REGEX.findall(text))

    def generate_temporal_questions(self, passage: str, num_questions: int = 5) -> List[str]:
        """Generate temporal-aware questions from a passage - matches notebook workflow"""
        
        # Extract years exactly like the notebook
        years = self.get_years_from_text(passage)
        if not years:
            # If no years, skip this passage (like in notebook)
            return []
        
        # Use first year like in the notebook
        first_year = sorted(list(years))[0]
        
        # Generate one question per passage (like notebook does)
        passages = [passage]
        year_batch = [first_year]
        
        # Generate questions using the batch function
        questions = self.generate_temporal_questions_batch(passages, year_batch)
        
        # If we need more questions, repeat with different years
        all_questions = []
        years_list = sorted(list(years))
        
        for i in range(num_questions):
            if i < len(questions) and questions[i]:
                all_questions.append(questions[i])
            elif years_list:
                # Generate with different year
                year_to_use = years_list[i % len(years_list)]
                additional_q = self.generate_temporal_questions_batch([passage], [year_to_use])
                if additional_q and additional_q[0]:
                    all_questions.append(additional_q[0])
        
        return all_questions[:num_questions]

    def process_dataset(self, dataset_path: str, output_path: str):
        """Process a dataset to generate questions"""
        logger.info(f"Processing dataset from {dataset_path}")
        
        # Load dataset (assuming it's in JSON format)
        if dataset_path.endswith('.json'):
            with open(dataset_path, 'r') as f:
                data = json.load(f)
        else:
            # Try to load as HuggingFace dataset
            data = load_dataset(dataset_path, split='train')
        
        generated_data = []
        
        for idx, item in enumerate(tqdm(data, desc="Generating questions")):
            if isinstance(item, dict):
                passage = item.get('text', item.get('passage', ''))
                passage_id = item.get('id', f'passage_{idx}')
            else:
                passage = str(item)
                passage_id = f'passage_{idx}'
            
            if not passage:
                continue
            
            # Generate questions
            questions = self.generate_temporal_questions(
                passage, 
                self.config['data']['generated_questions']['num_questions_per_passage']
            )
            
            for q_idx, question in enumerate(questions):
                generated_data.append({
                    'passage_id': passage_id,
                    'passage': passage,
                    'question': question,
                    'question_id': f"{passage_id}_q{q_idx}",
                    'temporal_type': 'generated'
                })
        
        # Save generated data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(generated_data, f, indent=2)
        
        logger.info(f"Generated {len(generated_data)} question-passage pairs")
        logger.info(f"Saved to {output_path}")
        
        return generated_data


def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    generator = T5QuestionGenerator(config)
    
    # Load FineWeb passages (matches notebook approach)
    fineweb_path = os.path.join(config['data']['fineweb']['output_path'], 'fineweb_passages.json')
    
    if os.path.exists(fineweb_path):
        print(f"Loading FineWeb passages from {fineweb_path}")
        with open(fineweb_path, 'r') as f:
            fineweb_data = json.load(f)
        
        # Convert to notebook format (id, text, title) - use all available passages
        all_passages = [(item['id'], item['text'], item['title']) for item in fineweb_data]
        
        # Sample for question generation like the notebook (NUM_QG_PASSAGES = 15000)
        num_qg_passages = min(15000, len(all_passages))  # notebook constant
        if len(all_passages) > num_qg_passages:
            import random
            random.seed(42)  # For reproducibility
            sample_passages = random.sample(all_passages, num_qg_passages)
            print(f"Sampled {len(sample_passages)} passages from {len(all_passages)} total FineWeb passages for question generation")
        else:
            sample_passages = all_passages
            print(f"Using all {len(sample_passages)} FineWeb passages for question generation")
    else:
        print(f"FineWeb data not found at {fineweb_path}, using sample passages...")
        # Fallback to hardcoded samples if FineWeb not available
        sample_passages = [
            (0, "The Renaissance began in Italy during the 14th century and lasted until the 17th century. It was characterized by a revival of classical learning and art.", "Renaissance"),
            (1, "World War II started on September 1, 1939, when Germany invaded Poland. The war lasted until September 2, 1945.", "WWII"),
            (2, "The Industrial Revolution began in Britain in the late 18th century and spread throughout Europe and North America during the 19th century.", "Industrial Revolution"),
            (3, "The American Civil War was fought from 1861 to 1865 between the northern and southern states.", "Civil War"),
            (4, "The Great Depression began in 1929 with the stock market crash and lasted throughout the 1930s.", "Great Depression"),
            (5, "The Cold War started in 1947 and lasted until 1991, representing geopolitical tension.", "Cold War")
        ]
    
    output_dir = config['data']['generated_questions']['output_path']
    os.makedirs(output_dir, exist_ok=True)
    
    # Process exactly like the notebook with batching
    synthetic_pairs = []  # (question, passage_text, passage_id)
    passage_batch, passage_info, year_batch = [], [], []
    QG_BATCH_SIZE = 8  # Match notebook batch size
    
    logger.info(f"Generating synthetic temporal questions from {len(sample_passages)} passages...")
    
    for (pid, text, title) in tqdm(sample_passages, desc="Processing passages"):
        years = generator.get_years_from_text(text)
        if not years:
            continue
        
        # Use first year like notebook
        first_year = sorted(list(years))[0]
        passage_batch.append(text)
        year_batch.append(first_year)
        passage_info.append((pid, text))
        
        if len(passage_batch) >= QG_BATCH_SIZE:
            # Process batch
            generated_questions = generator.generate_temporal_questions_batch(passage_batch, year_batch)
            for i, q in enumerate(generated_questions):
                if q:
                    p_id, p_text = passage_info[i]
                    synthetic_pairs.append((q, p_text, p_id))
            
            # Reset batch
            passage_batch, passage_info, year_batch = [], [], []
    
    # Process remaining passages
    if passage_batch:
        generated_questions = generator.generate_temporal_questions_batch(passage_batch, year_batch)
        for i, q in enumerate(generated_questions):
            if q:
                p_id, p_text = passage_info[i]
                synthetic_pairs.append((q, p_text, p_id))
    
    # Convert to the expected format
    all_generated = []
    for i, (question, passage, passage_id) in enumerate(synthetic_pairs):
        all_generated.append({
            'passage_id': f'sample_{passage_id}',
            'passage': passage,
            'question': question,
            'question_id': f'sample_{passage_id}_q{i}',
            'temporal_type': 'generated'
        })
    
    # Save sample generated data
    output_path = os.path.join(output_dir, 'sample_generated_questions.json')
    with open(output_path, 'w') as f:
        json.dump(all_generated, f, indent=2)
    
    logger.info(f"Generated sample questions saved to {output_path}")


if __name__ == "__main__":
    main()