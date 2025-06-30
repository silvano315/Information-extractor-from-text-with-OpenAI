from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Optional
import json

from .prompts import PromptTemplates

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_article_information(
    article_text: str,
    model: str = "gpt-4o-mini",
    use_structured_output: bool = True
) -> dict:
    """
    Extract people, roles, topic, subtopic, and date from article text using LLM.
    """

    try:
        messages = [
            {"role": "system", "content": PromptTemplates.SYSTEM_PROMPT},
            {"role": "user", "content": PromptTemplates.USER_PROMPT_TEMPLATE.format(text=article_text)}
        ]
        
        if use_structured_output:
            # Structured output using Pydantic validation
            from .schema import ArticleExtraction
            
            response = client.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=ArticleExtraction
            )
            
            # Parsed object as dict
            parsed_result = response.choices[0].message.parsed
            return {
                "success": True,
                "data": parsed_result.model_dump(),
                "metadata": {
                    "model": model,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            }
            
        else:
            # JSON mode fallback
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            result = json.loads(response.choices[0].message.content)
            return {
                "success": True,
                "data": result,
                "metadata": {
                    "model": model,
                    "tokens_used": response.usage.total_tokens if response.usage else 0
                }
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "data": None,
            "metadata": {}
        }
    
def process_single_article(article: dict, model: str = "gpt-4o-mini") -> dict:
    """
    Process a single article and return extraction results with metadata.
    """
    extraction_result = extract_article_information(article["text"], model)
    
    return {
        "article_id": article["id"],
        "extraction": extraction_result["data"] if extraction_result["success"] else None,
        "success": extraction_result["success"],
        "error": extraction_result.get("error"),
        "metadata": extraction_result["metadata"]
    }