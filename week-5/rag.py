import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = "nvidia/nemotron-3.5-lightning"

DOCS = [
    "The Denver Nuggets won the 2023 NBA Finals, beating the Miami Heat 4-1. Nikola Jokic was Finals MVP.",
    "The Boston Celtics won the 2024 NBA Finals, beating the Dallas Mavericks 4-1. Jaylen Brown was Finals MVP.",
    "The Oklahoma City Thunder won the 2025 NBA Finals, beating the Indiana Pacers 4-3. Shai Gilgeous-Alexander was Finals MVP.",
]

QUESTIONS = [
    "Who won the 2025 NBA Finals, and who was Finals MVP?",
    "Are the Denver Nuggets the defending NBA champions after winning the 2024 title?",
    "Who won the 2026 NBA Finals?",
    "Who do you think will win the 2027 NBA Finals?",
]

def retrieve(question):
    q_words = set(question.lower().split())
    return max(range(len(DOCS)), key=lambda i: len(q_words & set(DOCS[i].lower().split())))

def ask(prompt):
    r = client.chat.completions.create(model=MODEL, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"reasoning": {"enabled": False}})
    return r.choices[0].message.content

results = []
for q in QUESTIONS:
    name = retrieve(q)
    answer = ask(f"Use only this information to answer. If it doesn't say, answer \"I don't know\".\n\n"
                 f"Information: {DOCS[name]}\n\nQuestion: {q}")

    print(f"\n=== {q}\n[Retrieved] {name}\n[Answer] {answer}")
    results.append({"question": q, "retrieved": name, "answer": answer})

json.dump(results, open("week-5/outputs.json", "w"), indent=2)
print("\nSaved to week-5/outputs.json")