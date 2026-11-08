import re, json
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge

cats = ["alt.atheism", "soc.religion.christian"]
train = fetch_20newsgroups(subset="train", categories=cats)
test = fetch_20newsgroups(subset="test", categories=cats)

vec = TfidfVectorizer(lowercase=False)
model = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=0)
model.fit(vec.fit_transform(train.data), train.target)

accuracy = model.score(vec.transform(test.data), test.target)
print(f"Black-box accuracy on test set: {accuracy:.1%}")

def predict(texts):
    return model.predict_proba(vec.transform(texts))[:, 1]

def lime(text, num_samples=5000, num_features=6):
    words = re.findall(r"\w+", text)
    vocab = sorted(set(words))
    d = len(vocab)
    pos = {w: i for i, w in enumerate(vocab)}

    Z = np.random.randint(0, 2, size=(num_samples, d))
    Z[0] = 1
    texts = [" ".join(w for w in words if z[pos[w]]) for z in Z]

    probs = predict(texts)

    similarity = np.sqrt(Z.sum(axis=1) / d)
    distance = 1 - similarity
    weights = np.exp(-(distance ** 2) / 0.25 ** 2)

    simple = Ridge(alpha=1.0)
    simple.fit(Z, probs, sample_weight=weights)

    top = np.argsort(-np.abs(simple.coef_))[:num_features]
    return [(vocab[i], round(float(simple.coef_[i]), 3)) for i in top]

def header_only_words(text):
    header, _, body = text.partition("\n\n")
    return set(re.findall(r"\w+", header)) - set(re.findall(r"\w+", body))

np.random.seed(0)
N = 20
results, posts_with_header = [], 0

for i in range(N):
    text, true = test.data[i], cats[test.target[i]]
    p = predict([text])[0]
    explanation = lime(text)
    header = header_only_words(text)
    uses_header = any(word in header for word, _ in explanation)
    posts_with_header += uses_header

    if i < 3:
        print(f"\n=== Post {i}  (true: {true}, model says Christianity with p={p:.2f})")
        for word, w in explanation:
            tag = "  <- header only" if word in header else ""
            print(f"  {word:15s} {w:+.3f}  -> {'Christianity' if w > 0 else 'Atheism'}{tag}")

    results.append({"post": i, "true_label": true, "p_christianity": round(float(p), 3),
                    "explanation": explanation, "uses_header_word": uses_header})

print(f"\n{posts_with_header} of {N} posts ({posts_with_header / N:.0%}) have a "
      f"header-only word in their top 6.")

json.dump({"accuracy": accuracy, "posts_with_header_word": posts_with_header,
           "posts_checked": N, "results": results},
          open("week-6/outputs.json", "w"), indent=2)
print("Saved to week-6/outputs.json")