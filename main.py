import json
import random
from src.io.load_data import load_articles_json
from src.llm.openai_client import extract_article_information

def test_random_article(filepath: str = "data/raw/clean_articles.json"):
    """
    Load articles, pick a random one, and test extraction.
    """
    print("Loading articles...")
    articles = load_articles_json(filepath)
    print(f"Loaded {len(articles)} articles")
    
    # random article
    random_article = random.choice(articles)
    article_id = random_article["id"]
    article_text = random_article["text"]
    
    print(f"\nTesting article ID: {article_id[:8]}...")    
    print(f"\nArticle preview:")
    print("-" * 50)
    print(article_text[:300] + "..." if len(article_text) > 300 else article_text)
    print("-" * 50)
    
    print(f"\n🤖 Extracting information...")
    result = extract_article_information(article_text)
    
    if result["success"]:
        print("Extraction successful!")
        print(f"Tokens used: {result['metadata']['tokens_used']}")
        
        data = result["data"]
        print(f"\nPeople found: {len(data['people'])}")
        for i, person in enumerate(data["people"]):
            print(f"  {i+1} {person['name']}: {', '.join(person['roles'])}")
        
        print(f"\nTopic: {data['topic']}")
        print(f"  Subtopic: {data['subtopic']}")
        print(f"  Date: {data['date']}")
        
        print(f"\nFull JSON output:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
    else:
        print("Extraction failed!")
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    print("Article Extraction Test")
    
    try:
        print("\nTesting random article from dataset:")
        test_random_article()
    except Exception as e:
        print(f"Error testing random article: {e}")
    
    print("\nTest completed!")