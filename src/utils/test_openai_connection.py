from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def test_openai_connection():
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Sei un assistente utile."},
                {"role": "user", "content": "Test connessione. Rispondi: OK"}
            ],
            temperature=0
        )
        print("Connessione OK:", response.choices[0].message.content)
        return True
    except Exception as e:
        print("Errore:", e)
        return False

test_openai_connection()