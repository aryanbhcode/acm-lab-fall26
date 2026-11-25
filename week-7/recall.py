import json, os
import numpy as np
import scipy.sparse as sp
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

OUT = os.path.dirname(os.path.abspath(__file__))
K = 10
NUM_SAMPLES = 5000
NUM_DOCS = 100
SEED = 0

PAPER = {
    "Books": {
        "Sparse LR": {"Random": 17.4, "Greedy": 64.3, "LIME": 92.1},
        "Decision Tree": {"Random": 20.6, "Greedy": 37.0, "LIME": 97.0},
    },
    "DVDs": {
        "Sparse LR": {"Random": 19.2, "Greedy": 63.4, "LIME": 90.2},
        "Decision Tree": {"Random": 17.4, "Greedy": 47.6, "LIME": 97.8},
    },
}

cats = ["alt.atheism", "soc.religion.christian"]
train = fetch_20newsgroups(subset="train", categories=cats)
test = fetch_20newsgroups(subset="test", categories=cats)

vec = CountVectorizer(binary=True)
Xtr = vec.fit_transform(train.data)
Xte = vec.transform(test.data)
V = Xtr.shape[1]
vocab = vec.get_feature_names_out()


def l1_logreg(C):
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, random_state=SEED)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=SEED)


def train_sparse_lr():
    best = l1_logreg(1e-3).fit(Xtr, train.target)
    for C in np.logspace(-3, 1, 40):
        m = l1_logreg(C).fit(Xtr, train.target)
        if np.count_nonzero(m.coef_) <= K:
            best = m
    return best


def train_tree():
    return DecisionTreeClassifier(max_depth=K, random_state=SEED).fit(Xtr, train.target)


def gold_lr(model, idx):
    nz = set(np.flatnonzero(model.coef_[0]))
    return {j for j in idx if j in nz}


def gold_tree(model, row):
    path = model.decision_path(row).indices
    feats = model.tree_.feature[path]
    present = set(row.indices)
    return {int(f) for f in feats if f >= 0 and int(f) in present}


def to_matrix(Z, idx):
    rows, cols = np.nonzero(Z)
    return sp.csr_matrix((np.ones(len(rows)), (rows, idx[cols])), shape=(Z.shape[0], V))


def lime(model, idx, label, rng):
    d = len(idx)
    Z = rng.integers(0, 2, size=(NUM_SAMPLES, d))
    Z[0] = 1
    y = model.predict_proba(to_matrix(Z, idx))[:, label]
    similarity = np.sqrt(Z.sum(axis=1) / d)
    distance = 1 - similarity
    weights = np.exp(-(distance ** 2) / 0.25 ** 2)
    simple = Ridge(alpha=0.01).fit(Z, y, sample_weight=weights)
    top = np.argsort(-np.abs(simple.coef_))[:K]
    return {int(idx[i]) for i in top}


def random_pick(idx, rng):
    return {int(j) for j in rng.choice(idx, size=min(K, len(idx)), replace=False)}


def greedy(model, idx, label):
    d = len(idx)
    keep = np.ones(d, dtype=int)
    removed = []
    for _ in range(min(K, d)):
        candidates = np.flatnonzero(keep)
        if len(candidates) == 0:
            break
        Z = np.tile(keep, (len(candidates), 1))
        Z[np.arange(len(candidates)), candidates] = 0
        p = model.predict_proba(to_matrix(Z, idx))[:, label]
        best = candidates[int(np.argmin(p))]
        keep[best] = 0
        removed.append(int(idx[best]))
        if model.predict_proba(to_matrix(keep[None, :], idx))[0].argmax() != label:
            break
    return set(removed)


def run(name, model, gold_fn):
    rng = np.random.default_rng(SEED)
    scores = {"Random": [], "Greedy": [], "LIME": []}
    used = 0
    for i in range(Xte.shape[0]):
        if used >= NUM_DOCS:
            break
        row = Xte[i]
        idx = np.array(sorted(row.indices))
        gold = gold_fn(model, idx, row)
        if not gold or len(idx) < 2:
            continue
        label = int(model.predict(row)[0])
        for method, found in (
            ("Random", random_pick(idx, rng)),
            ("Greedy", greedy(model, idx, label)),
            ("LIME", lime(model, idx, label, rng)),
        ):
            scores[method].append(len(found & gold) / len(gold))
        used += 1
    acc = model.score(Xte, test.target)
    print(f"\n{name}: test accuracy {acc:.1%}, {used} documents explained")
    return {m: 100 * float(np.mean(v)) for m, v in scores.items()}, acc


lr = train_sparse_lr()
tree = train_tree()
print("Sparse LR uses these words:", ", ".join(vocab[np.flatnonzero(lr.coef_[0])]))

ours = {}
ours["Sparse LR"], acc_lr = run("Sparse LR", lr, lambda m, idx, row: gold_lr(m, idx))
ours["Decision Tree"], acc_tree = run("Decision Tree", tree, lambda m, idx, row: gold_tree(m, row))

methods = ["Random", "Greedy", "LIME"]
print("\nRecall of truly important words (%), top 10 words per explanation")
print(f"{'':16}{'':9}" + "".join(f"{m:>9}" for m in methods))
table = {}
for model_name in ["Sparse LR", "Decision Tree"]:
    rows = [("Mine (20NG)", ours[model_name]),
            ("Paper Books", PAPER["Books"][model_name]),
            ("Paper DVDs", PAPER["DVDs"][model_name])]
    for label, vals in rows:
        print(f"{model_name:16}{label:<13}" + "".join(f"{vals[m]:9.1f}" for m in methods))
    table[model_name] = {label: vals for label, vals in rows}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
width = 0.26
for ax, model_name in zip(axes, ["Sparse LR", "Decision Tree"]):
    x = np.arange(len(methods))
    for k, (label, color) in enumerate([("Mine (20NG)", "#1f77b4"),
                                        ("Paper Books", "#aec7e8"),
                                        ("Paper DVDs", "#c7c7c7")]):
        vals = [table[model_name][label][m] for m in methods]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=label, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_title(model_name)
    ax.set_ylim(0, 110)
axes[0].set_ylabel("Recall of truly important words (%)")
axes[1].legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "recall.png"), dpi=150)

with open(os.path.join(OUT, "outputs.json"), "w") as f:
    json.dump({"k": K, "num_samples": NUM_SAMPLES, "num_docs": NUM_DOCS,
               "test_accuracy": {"Sparse LR": acc_lr, "Decision Tree": acc_tree},
               "recall": table}, f, indent=2)
print("\nSaved week-7/recall.png and week-7/outputs.json")