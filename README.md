# GPT-STRUCTURED-INFO-EXTRACTION

*From Unstructured Text to JSON-Structured Insights*

![last commit](https://img.shields.io/badge/last%20commit-today-blue)
![python](https://img.shields.io/badge/python-100%25-blue)
![openai](https://img.shields.io/badge/OpenAI-API-green)

---

## Approach

### Pipeline Architecture

After carefully analyzing the project requirements, I decided to implement a task-oriented approach for this information extraction project using OpenAI's LLM family. The goal was to develop both a [notebook](Test_Pratico_Colloquio_Rogue_Waves_AI_Silvano_Quarto.ipynb) for interactive development and testing of different pipeline phases, and a [main script](main.py) for terminal-based execution (excluding evaluation).
My approach can be divided into several key phases:

1. **Data Loading**: Load articles from JSON format
2. **Articles Exploration**: A small EDA implementation to evaluate basic statistics
2. **Preprocessing**:  Minimal but effective text cleaning focused on markdown removal while preserving semantic content. (I decided not to use it for the following steps due to time)
3. **LLM Extraction**: Single-step extraction using OpenAI's structured output
4. **Validation**: Pydantic schema validation for output consistency
5. **Evaluation**: Precision/Recall/F1 metrics against ground truth

You can find more details about my decisions and project details in the notebook.

### Design Decisions
- Process all required information in one API call for cost efficiency
- Use OpenAI's `response_format` with Pydantic schemas for JSON generation
- Process only articles with corresponding ground truth annotations (200 articles)

---

## Models Used

I Chose gpt-4o-mini to process 200+ articles within budget constraints while maintaining high extraction quality. This decision was due to the choice of optimizing costs and avoiding possible risks of unexpected errors. In addition, the results obtained seemed valid enough to continue with the following phases.

---

## Preprocessing Steps

My initial idea was to do a more completed preprocessing step, but in accordance with the tight schedule and evaluating the data quite clean and valid I preferred to continue the other phases of the project. Only the initial markdown is removed here.

---

## Installation & Setup

### Requirements
```bash
git clone https://github.com/silvano315/Information-extractor-from-text-with-OpenAI.git
```

```bash
pip install -r requirements.txt
```

### Environment Setup
1. Create `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

2. Create directory structure:
```bash
mkdir -p data/{raw,preprocessed,output}
```

---

## Usage

### Quick Test
Test extraction on a single random article:
```bash
python main.py
```

---

## Evaluation Metrics

You can find evaluation results here or you can retrive them using the [notebook](Test_Pratico_Colloquio_Rogue_Waves_AI_Silvano_Quarto.ipynb).

| Metric | Description | Score |
|--------|-------------|-------|
| **Entity F1** | Named entity extraction accuracy | 0.812 |
| **Role F1** | Role assignment accuracy | 0.741 |
| **Topic Accuracy** | Topic classification accuracy | 0.975 |
| **Subtopic Accuracy** | Subtopic classification accuracy | 0.970 |

The final result I think is quite satisfactory in terms of both performance and realization. Interesting result for topics and subtopics that achieve very high accuracy. This result shows the strength of OpenAI models when a well-structured schema of possible options is offered.

Entity extraction (for entities and roles) shows a good F1 scores, despite the complexity for named entity recognition tasks. I used F1 score for these tasks because I learnt that it should be a standard metric for NLP and NER. The lower F1 compared to accuracy is expected because of open-vocabulary extraction. Role assignment scores slightly lower due to possible semantic variations. 