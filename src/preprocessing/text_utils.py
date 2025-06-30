
import json
import re
import os
from typing import List, Dict, Any
from pathlib import Path

def preprocess_article(text: str) -> str:
    """
    Apply minimal preprocessing to article text.
    """
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    return text.strip()

def preprocess_articles_dataset(
    input_filepath: str, 
    output_dir: str = "data/preprocessed"
) -> str:
    """
    Apply preprocessing to all articles and save to new JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"Loaded {len(articles)} articles from {input_filepath}")
    
    # Apply preprocessing
    preprocessed_articles = []
    for article in articles:
        preprocessed_article = {
            "id": article["id"],
            "text": preprocess_article(article["text"])
        }
        preprocessed_articles.append(preprocessed_article)
    
    input_filename = Path(input_filepath).stem
    output_filepath = os.path.join(output_dir, f"{input_filename}_preprocessed.json")
    
    # Save preprocessed articles
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_articles, f, indent=2, ensure_ascii=False)
    
    print(f"Total articles processed: {len(preprocessed_articles)}")
    
    return output_filepath