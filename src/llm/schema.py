from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class Person(BaseModel):
    """Individual person with their associated roles."""
    name: str = Field(..., description="Full name of the person")
    roles: List[str] = Field(..., description="List of roles/professions for this person")

class ArticleExtraction(BaseModel):
    """Complete extraction results for a single article."""
    people: List[Person] = Field(..., description="List of people mentioned in the article")
    topic: str = Field(..., description="Main topic category")
    subtopic: str = Field(..., description="Specific subtopic within the topic")
    date: str = Field(..., description="Article date in YYYY-MM-DD format")

class ExtractionResult(BaseModel):
    """Complete extraction result with metadata."""
    article_id: str = Field(..., description="Unique identifier for the article")
    extraction: ArticleExtraction = Field(..., description="Extracted information")
    metadata: Optional[dict] = Field(default_factory=dict, description="Processing metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "article_id": "0142e7ab-2349-4098-9c74-59b01adc53b5",
                "extraction": {
                    "people": [
                        {
                            "name": "Lidia Monteverdi-Gremese",
                            "roles": ["Journalist", "Reporter"]
                        },
                        {
                            "name": "Fabio Battelli",
                            "roles": ["Caregiver", "Patient"]
                        }
                    ],
                    "topic": "Health",
                    "subtopic": "Epidemic",
                    "date": "2025-05-11"
                },
                "metadata": {
                    "model_used": "gpt-4o-mini",
                    "processing_time": 2.3,
                    "tokens_used": 150
                }
            }
        }

