import json
from typing import List, Dict, Set
from src.io.load_data import load_articles_json, load_ground_truth_json
from src.llm.openai_client import extract_article_information

def extract_ground_truth_uuids(ground_truth_data: List[Dict]) -> Set[str]:
    """
    Extract all UUIDs from ground truth data.
    """
    uuids = {item["uuid"] for item in ground_truth_data if "uuid" in item}
    print(f"Found {len(uuids)} UUIDs in ground truth")
    return uuids

def match_articles_with_ground_truth(
    articles: List[Dict], 
    ground_truth_uuids: Set[str]
) -> List[Dict]:
    """
    Filter articles to only those present in ground truth.
    """
    matched_articles = []
    
    for article in articles:
        article_id = article["id"]
        # Check if article ID matches any UUID in ground truth
        if article_id in ground_truth_uuids:
            matched_articles.append(article)
    
    print(f"Matched {len(matched_articles)} articles with ground truth")
    print(f"Coverage: {len(matched_articles)}/{len(ground_truth_uuids)} ground truth entries")
    
    return matched_articles

def process_ground_truth_articles(
    articles_filepath: str,
    ground_truth_filepath: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 10
) -> List[Dict]:
    """
    Process only articles that have ground truth annotations.
    """
    print("Starting Ground Truth Article Processing")    
    print("Loading datasets...")
    articles = load_articles_json(articles_filepath)
    ground_truth = load_ground_truth_json(ground_truth_filepath)
    
    # Extract UUIDs from ground truth
    gt_uuids = extract_ground_truth_uuids(ground_truth)
    
    # Match articles with ground truth
    matched_articles = match_articles_with_ground_truth(articles, gt_uuids)
    
    if not matched_articles:
        print("No articles matched with ground truth!")
        return []
    
    # Process matched articles
    print(f"\nProcessing {len(matched_articles)} articles...")
    results = []
    
    for i, article in enumerate(matched_articles):
        print(f"Processing {i+1}/{len(matched_articles)} - ID: {article['id'][:8]}...")
        
        extraction_result = extract_article_information(article["text"], model)
        
        result = {
            "article_id": article["id"],
            "extraction": extraction_result["data"] if extraction_result["success"] else None,
            "success": extraction_result["success"],
            "error": extraction_result.get("error"),
            "metadata": extraction_result["metadata"]
        }
        
        results.append(result)
        
        # Save intermediate results every batch_size articles
        if (i + 1) % batch_size == 0:
            save_intermediate_results(results, f"intermediate_results_batch_{i+1}.json")
            print(f"Saved intermediate results (batch {i+1})")
    
    # Summary
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful
    
    print(f"\nProcessing Summary:")
    print(f"   - Successful: {successful}/{len(results)}")
    print(f"   - Failed: {failed}/{len(results)}")
    
    return results

def save_intermediate_results(results: List[Dict], filename: str) -> None:
    """
    Save intermediate processing results.
    """
    with open(f"data/output/{filename}", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def save_final_results(results: List[Dict], output_filepath: str = "data/output/extraction_results.json") -> None:
    """
    Save final extraction results to JSON file.
    """
    import os
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Final results saved to: {output_filepath}")