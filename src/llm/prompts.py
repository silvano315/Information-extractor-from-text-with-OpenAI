class PromptTemplates:
    SYSTEM_PROMPT = """
You are an expert AI for extracting structured information from news articles.
Extract people with their roles, classify topic/subtopic, and find the article date.
Respond only in JSON format, without additional explanations.
If information is not present, use empty lists or appropriate defaults.
"""

    USER_PROMPT_TEMPLATE = """
Analyze the following news article and extract the required information.

ARTICLE TEXT:
{text}

EXTRACT:
1. People mentioned with their roles/professions
2. Main topic category 
3. Specific subtopic
4. Article date

VALID TOPICS AND SUBTOPICS:
- Politics: Election, Policy, Corruption, Diplomacy
- Sports: Football, Olympics, Doping, Injury  
- Crime: Robbery, Murder, Fraud, Drug Trafficking
- Economy: Inflation, Stock Market, Unemployment, GDP
- Environment: Climate Change, Pollution, Wildlife, Natural Disaster
- Culture: Festival, Cinema, Literature, Art Exhibition
- Science: Astronomy, Physics, Biology, Research Discovery
- Technology: AI, Cybersecurity, Gadgets, Software
- Health: Epidemic, Vaccination, Nutrition, Mental Health

REQUIRED JSON FORMAT:
{{
  "people": [
    {{
      "name": "Full Name",
      "roles": ["Role1", "Role2"]
    }}
  ],
  "topic": "TopicName",
  "subtopic": "SubtopicName", 
  "date": "YYYY-MM-DD"
}}
"""