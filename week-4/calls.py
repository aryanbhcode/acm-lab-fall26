import os, json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"])

MODELS = ["google/gemma-4-31b-it",
          "nvidia/nemotron-3.5-lightning"]     
PROMPT = "In two sentences, who do you think will win the 2026 NBA Finals and why?"

results = []
for m in MODELS:
    r = client.chat.completions.create(model=m, max_tokens=500,
        messages=[{"role": "user", "content": PROMPT}])
    results.append({"model": m, "output": r.choices[0].message.content,
                    "tokens": r.usage.total_tokens})
    print(f"--- {m}\n{r.choices[0].message.content}\n")

json.dump(results, open("week-4/outputs.json", "w"), indent=2)