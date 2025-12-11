"""
FineWeb Dataset Loader for Time-Aware RAG
Downloads and processes FineWeb-edu data to collect temporal passages
Exactly matches the notebook implementation
"""

import os
import re
import json
import yaml
from typing import List, Tuple, Set
from tqdm import tqdm
from datasets import load_dataset

# Use same regex as notebook
YEAR_REGEX = re.compile(r"\b(18[0-9]{2}|19[0-9]{2}|20[0-2][0-9])\b")


def _norm(s: str) -> str:
    """Normalize text for deduplication - matches notebook"""
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def get_years_from_text(text: str) -> Set[str]:
    """Extract years from text - matches notebook"""
    return set(YEAR_REGEX.findall(text))


def load_fineweb_dataset(config: dict) -> List[Tuple[int, str, str]]:
    """
    Load FineWeb dataset exactly matching notebook implementation
    Returns list of (id, text, title) tuples
    """
    fineweb_config = config['data']['fineweb']
    
    print("\n--- Step 4: Preparing FineWeb Data ---")
    print(f"Streaming {fineweb_config['dataset_name']} ({fineweb_config['config_name']})...")
    
    # Load dataset exactly like notebook
    dataset_stream = load_dataset(
        fineweb_config['dataset_name'],
        name=fineweb_config['config_name'],
        split=fineweb_config['split'],
        streaming=True,
    )
    
    train_passages_all = []
    seen_texts = set()
    current_id = 0
    
    sample_size = fineweb_config['sample_size']
    max_chars = fineweb_config['max_passage_chars']
    
    print(f"Filtering stream for passages containing years (1900-2029)...")
    pbar = tqdm(total=sample_size, desc="Collecting Passages")
    
    for row in dataset_stream:
        if len(train_passages_all) >= sample_size:
            break
            
        raw_text = row.get('text', "")
        if not raw_text:
            continue
            
        # Truncate text exactly like notebook
        text_slice = raw_text[:max_chars]
        
        # Skip if no years found
        if not get_years_from_text(text_slice):
            continue
            
        # Deduplication exactly like notebook
        norm_text = _norm(text_slice[:100])
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)
        
        # Add to collection exactly like notebook format
        train_passages_all.append((current_id, text_slice, f"fineweb_{current_id}"))
        current_id += 1
        pbar.update(1)
    
    pbar.close()
    print(f"Clean training set size (FineWeb): {len(train_passages_all)}")
    
    return train_passages_all


def save_fineweb_data(passages: List[Tuple[int, str, str]], output_path: str):
    """Save FineWeb data to JSON format for later use"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to JSON serializable format
    data = []
    for passage_id, text, title in passages:
        data.append({
            'id': passage_id,
            'text': text, 
            'title': title,
            'source': 'fineweb-edu'
        })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved {len(data)} FineWeb passages to {output_path}")


def main():
    """Main function to download and process FineWeb data"""
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = config['data']['fineweb']['output_path']
    os.makedirs(output_dir, exist_ok=True)
    
    # Load FineWeb data
    print("Starting FineWeb dataset download and processing...")
    passages = load_fineweb_dataset(config)
    
    if not passages:
        print("ERROR: No passages collected from FineWeb!")
        return False
    
    # Save processed data
    output_path = os.path.join(output_dir, 'fineweb_passages.json')
    save_fineweb_data(passages, output_path)
    
    print(f"\nFineWeb processing complete!")
    print(f"Collected {len(passages)} temporal passages")
    print(f"Saved to: {output_path}")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)