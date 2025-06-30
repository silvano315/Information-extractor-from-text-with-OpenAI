import random
from typing import Dict, List, Tuple
import tiktoken
import re

class EDA:
    def __init__(self, data: List[Dict[str, str]]):
        """
        Initialize the EDA instance with list of articles.
        """
        self.data = data
        self.texts = [article['text'] for article in data]
        self.ids = [article['id'] for article in data]
        
    def count_documents(self) -> int:
        """Return total number of documents."""
        return len(self.data)
    
    def text_length_stats(self) -> Dict[str, int]:
        """Calculate basic text length statistics."""
        lengths = [len(text) for text in self.texts]
        word_counts = [len(text.split()) for text in self.texts]
        
        return {
            "num_documents": len(self.data),
            "avg_length_chars": int(sum(lengths) / len(lengths)),
            "min_length_chars": min(lengths),
            "max_length_chars": max(lengths),
            "avg_words": int(sum(word_counts) / len(word_counts)),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
        }
    
    def token_stats(self, model: str = "gpt-4o-mini") -> Dict[str, int]:
        """Calculate token statistics for the given model."""
        encoding = tiktoken.encoding_for_model(model)
        token_counts = [len(encoding.encode(text)) for text in self.texts]
        
        return {
            "avg_tokens": int(sum(token_counts) / len(token_counts)),
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "total_tokens": sum(token_counts)
        }
    
    def count_tokens(self, text: str, model: str = "gpt-4o-mini") -> int:
        """Count tokens in a single text."""
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    
    def preview_articles(self, n: int = 3, preview_chars: int = 200) -> None:
        """Print preview of first n articles."""
        for i, article in enumerate(self.data[:n]):
            print(f"\n--- Article {i+1} (ID: {article['id'][:8]}...) ---")
            print(f"Length: {len(article['text'])} chars, {len(article['text'].split())} words")
            print(f"Preview: {article['text'][:preview_chars]}...")
            print("-" * 50)