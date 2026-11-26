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
NUM_DOCS = 60
SEEDS = [0, 1, 2]
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 2500, 5000]
MODEL_SEED = 0

lines = []


def say(text=""):
    print(text)
    lines.append(text)


cats = ["alt.atheism", "soc.religion.christian"]
train = fetch_20newsgroups(subset="train", categories=cats)
test = fetch_20newsgroups(subset="test", categories=cats)
vec = CountVectorizer(binary=True)
Xtr = vec.fit_transform(train.data)
Xte = vec.transform(test.data)
V = Xtr.shape[1]


def l1_logreg(C):
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) >= (1, 8):
        return LogisticRegression(l1_ratio=1, solver="liblinear", C=C, random_state=MODEL_SEED)
    return LogisticRegression(penalty="l1", solver="liblinear", C=C, random_state=MODEL_SEED)


def train_sparse_lr():
    best = l1_logreg(1e-3).fit(Xtr, train.target)
    for C in np.logspace(-3, 1, 40):
        m = l1_logreg(C).fit(Xtr, train.target)
        if np.count_nonzero(m.coef_) <= K:
            best = m
    return best


def train_tree():
    return DecisionTreeClassifier(max_depth=K, random_state=MODEL_SEED).fit(Xtr, train.target)


def gold_lr(model, idx, row):
    nz = set(np.flatnonzero(model.coef_[0]))
    return {int(j) for j in idx if j in nz}


def gold_tree(model, idx, row):
    path = model.decision_path(row).indices
    feats = model.tree_.feature[path]
    present = set(row.indices)
    return {int(f) for f in feats if f >= 0 and int(f) in present}


def to_matrix(Z, idx):
    rows, cols = np.nonzero(Z)
    return sp.csr_matrix((np.ones(len(rows)), (rows, idx[cols])), shape=(Z.shape[0], V))


def lime(model, idx, label, rng, n):
    d = len(idx)
    Z = rng.integers(0, 2, size=(n, d))
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


def collect(model, gold_fn):
    docs = []
    for i in range(Xte.shape[0]):
        if len(docs) >= NUM_DOCS:
            break
        row = Xte[i]
        idx = np.array(sorted(row.indices))
        gold = gold_fn(model, idx, row)
        if not gold or len(idx) < 2:
            continue
        docs.append((idx, gold, int(model.predict(row)[0])))
    return docs


def experiment(model, docs):
    lime_rec = np.zeros((len(SEEDS), len(SAMPLE_SIZES), len(docs)))
    rand_rec = np.zeros((len(SEEDS), len(docs)))
    greedy_rec = np.zeros(len(docs))
    for j, (idx, gold, label) in enumerate(docs):
        greedy_rec[j] = len(greedy(model, idx, label) & gold) / len(gold)
        for s, seed in enumerate(SEEDS):
            rng = np.random.default_rng(seed * 1000 + j)
            rand_rec[s, j] = len(random_pick(idx, rng) & gold) / len(gold)
            for k, n in enumerate(SAMPLE_SIZES):
                lime_rec[s, k, j] = len(lime(model, idx, label, rng, n) & gold) / len(gold)
    return 100 * lime_rec, 100 * rand_rec, 100 * greedy_rec


say(f"Extension: how many perturbed samples does LIME need?  (K = {K}, {NUM_DOCS} posts per model, seeds {SEEDS})")
lr = train_sparse_lr()
tree = train_tree()
models = {"Sparse LR": (lr, gold_lr), "Decision Tree": (tree, gold_tree)}

summary = {}
for name, (model, gold_fn) in models.items():
    docs = collect(model, gold_fn)
    lens = np.array([len(d[0]) for d in docs])
    gold_sizes = np.array([len(d[1]) for d in docs])
    lime_rec, rand_rec, greedy_rec = experiment(model, docs)

    per_seed = lime_rec.mean(axis=2)
    mean, std = per_seed.mean(axis=0), per_seed.std(axis=0)
    greedy_mean = float(greedy_rec.mean())
    random_mean = float(rand_rec.mean())

    say(f"\n=== {name}: {len(docs)} posts, avg {lens.mean():.0f} distinct words per post, "
        f"avg {gold_sizes.mean():.1f} true words per post")
    say("Samples".ljust(14) + "".join(f"{n:>10}" for n in SAMPLE_SIZES))
    say("LIME recall %".ljust(14) + "".join(f"{m:>10.1f}" for m in mean))
    say("  (std)".ljust(14) + "".join(f"{s:>10.1f}" for s in std))
    say(f"Greedy recall {greedy_mean:.1f}%   Random recall {random_mean:.1f}%  (these do not depend on samples)")

    groups = np.array_split(np.argsort(lens), 3)
    by_length = {}
    for label, g in zip(["shortest third", "middle third", "longest third"], groups):
        vals = lime_rec[:, :, g].mean(axis=(0, 2))
        by_length[label] = {"avg_words": float(lens[g].mean()), "recall": vals.tolist()}
        say(f"{label} ({lens[g].mean():.0f} words)".ljust(34) + "".join(f"{v:>10.1f}" for v in vals))

    enough = [n for n, m in zip(SAMPLE_SIZES, mean) if m >= 90 and m >= greedy_mean]
    say(f"Smallest sample size where LIME is at least 90% and at least Greedy: "
        f"{enough[0] if enough else 'none tested'}")

    summary[name] = {"samples": SAMPLE_SIZES, "lime_mean": mean.tolist(), "lime_std": std.tolist(),
                     "greedy": greedy_mean, "random": random_mean,
                     "avg_words_per_post": float(lens.mean()), "avg_true_words": float(gold_sizes.mean()),
                     "by_length": by_length}

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, name in zip(axes, models):
    s = summary[name]
    ax.errorbar(SAMPLE_SIZES, s["lime_mean"], yerr=s["lime_std"], marker="o", color="#1f77b4", label="LIME")
    ax.axhline(s["greedy"], linestyle="--", color="#ff7f0e", label="Greedy")
    ax.axhline(s["random"], linestyle=":", color="#7f7f7f", label="Random")
    ax.set_xscale("log")
    ax.set_xlabel("Perturbed samples per explanation")
    ax.set_title(name)
    ax.set_ylim(0, 105)
axes[0].set_ylabel("Recall of truly important words (%)")
axes[1].legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "samples_curve.png"), dpi=150)

with open(os.path.join(OUT, "outputs.json"), "w") as f:
    json.dump({"k": K, "num_docs": NUM_DOCS, "seeds": SEEDS, "results": summary}, f, indent=2)
with open(os.path.join(OUT, "run_output.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
say("\nSaved week-8/samples_curve.png, week-8/outputs.json, week-8/run_output.txt")