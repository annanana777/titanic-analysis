# Titanic Survival Analysis

A full-stack web application that analyzes the Titanic dataset using FastAPI, pandas, Plotly, and Claude AI.

## Project Structure

```
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── titanic.csv
├── big_data_analysis.ipynb
└── README.md
```

## Setup

Install dependencies:
```
pip install -r backend/requirements.txt
```

Set your Anthropic API key:
```
export ANTHROPIC_API_KEY="your-key-here"
```

Start the backend:
```
cd backend
uvicorn main:app --reload
```

Open `frontend/index.html` in your browser, upload `titanic.csv` and click Analyze.

## Technologies

- Frontend: HTML, CSS, JavaScript
- Backend: Python, FastAPI
- Data: pandas, numpy
- Charts: Plotly
- AI: Anthropic Claude API

## Dataset

891 passengers, 12 original columns. New columns added: Title, FamilySize, IsAlone, AgeGroup, PclassLabel, SurvivedLabel.

## Key Findings

- Overall survival rate: 38.4%
- Women survived at 74% vs men at 19%
- 1st class survival: 63% vs 3rd class: 24%
- Children had higher survival rate than adults
