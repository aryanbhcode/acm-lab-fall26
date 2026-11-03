# ACM Research Lab — Fall 2026

Aryan Bhatia's work for the ACM Research Lab, one folder per week.

## Setup

1. Create a virtual environment and install dependencies:

   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install openai python-dotenv
   ```

2. Create a `.env` file in the repo root with your OpenRouter key:

   ```
   OPENROUTER_API_KEY=your-key-here
   ```

   `.env` is listed in `.gitignore` and is never committed.

## Week 4: OpenRouter API calls

`week-4/calls.py` sends the same prompt to two models through OpenRouter,
one of them open-weight (Gemma 4 31B and Nemotron 3.5 Lightning), and saves
the answers to `week-4/outputs.json`.

Run from the repo root:

```
python3 week-4/calls.py
```

## Week 5: Basic RAG

`week-5/rag.py` answers NBA Finals questions with retrieval. It picks the
document that shares the most words with the question, gives it to
Nemotron 3.5 Lightning, and saves the answers to
`week-5/outputs.json`.

```
python3 week-5/rag.py
```
