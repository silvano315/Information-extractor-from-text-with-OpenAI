import json
from typing import List, Dict, Any
from pathlib import Path

def load_articles_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load articles from JSON file and return as list of dictionaries.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {filepath}")
    return articles

def load_ground_truth_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load ground truth evaluation data from JSON file.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    
    print(f"Loaded {len(ground_truth)} ground truth samples from {filepath}")
    return ground_truth