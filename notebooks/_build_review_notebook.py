"""Builds notebooks/selector.ipynb: the simple, review-ready version of the training notebook.

The previous long notebook is kept as notebooks/archive/selector_full.ipynb.
Run:  python notebooks/_build_review_notebook.py
"""
import os
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
cells = []


def md(s):
    cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    cells.append(nbf.v4.new_code_cell(s.strip("\n")))


# ---------------------------------------------------------------- title + story
md(r'''
# ML-guided selection of graph-query strategies in Apache Spark

**The problem in one line:** Spark can run our two-hop graph query, `(a)->(b)->(c)`, in three ways:
**native** (GraphFrames `find`), **edge-join** (a plain self-join) and **hub-split** (splits very busy "hub" vertices across a grid).
No single way is fastest for every graph. We train a model that looks at a graph's shape **before** running it and picks the fastest way.

This notebook follows the six-step data-visualisation story:

| Step | Data-visualisation process | What it means in our project | Where in this notebook |
|---|---|---|---|
| 1 | Identify data sources | 62 synthetic graphs (5 families + a hub sweep) | Step 1 |
| 2 | Extract data | Each graph is run on Spark with every strategy, and the times are recorded | Step 2 |
| 3 | Map data (format + clean) | 13 features per graph; label = the fastest strategy | Step 3 |
| 4 | Upload to one standard store | Everything goes into one table, `combined_table.csv` (62 rows) | Step 4 |
| 5 | Graphically represent | Train, validate and test the models; compare them with the baselines | Step 5 |
| 6 | Save and share reports | Results file, chart PNGs and a Tableau "decision explorer" | Step 6 |

Every chart has two short notes: **How we got it** and **What it tells us**.
Run it with *Kernel → Restart & Run All*. It takes about 2–3 minutes, and the last cell checks that the numbers are the same as in our report.
''')

code(r'''
# Setup: libraries, paths, fixed seeds (so every run gives the same numbers)
import os, sys, json, warnings
import numpy as np, pandas as pd
%matplotlib inline
import matplotlib, matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Wedge
from IPython.display import display, Markdown
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
warnings.filterwarnings("ignore")

def find_repo(start):
    d = os.path.abspath(start)
    while not os.path.exists(os.path.join(d, "datasets", "derived", "training", "combined_table.csv")):
        if os.path.dirname(d) == d:
            raise SystemExit("Open this notebook from inside the project folder.")
        d = os.path.dirname(d)
    return d

REPO = find_repo(os.getcwd()); sys.path.insert(0, REPO)
from ml.build_feature_table import FEATURE_COLUMNS          # the 11 features computed from each graph

TABLE = os.path.join(REPO, "datasets", "derived", "training", "combined_table.csv")
OUT = os.path.join(REPO, "datasets", "derived", "review"); os.makedirs(OUT, exist_ok=True)
SEED, NTREES = 0, 300
np.random.seed(SEED)

STRATS = ["edgejoin__aqe_off", "native__aqe_on", "hypercube_unified__aqe_off"]   # the 3 strategies
NAME = {"edgejoin__aqe_off": "edge-join", "native__aqe_on": "native", "hypercube_unified__aqe_off": "hub-split"}
CLASS_OF = {"edgejoin__aqe_off": "edgejoin", "native__aqe_on": "native", "hypercube_unified__aqe_off": "hubsplit"}
CLASSES = ["edgejoin", "hubsplit", "native"]
COL = {"edge-join": "#4C78A8", "native": "#72B7B2", "hub-split": "#E45756", "ML selector": "#2E8B57", "baseline": "#B0B0B0"}
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RES = {}
print("Project folder:", REPO)
''')

code(r'''
# The story in one picture
steps = [("1. Identify sources", "62 synthetic graphs\n5 families + hub sweep"),
         ("2. Extract", "Run each graph on Spark\nwith 3 strategies, time it"),
         ("3. Map / clean", "13 features per graph\nlabel = fastest strategy"),
         ("4. One standard table", "combined_table.csv\n62 rows x 45 columns"),
         ("5. Represent", "train, validate, test\ncompare with baselines"),
         ("6. Save & share", "results.json, PNG charts\nTableau decision explorer")]
fig, ax = plt.subplots(figsize=(13, 4.2)); ax.axis("off"); ax.set_xlim(0, 3); ax.set_ylim(0, 2)
pos = [(0, 1), (1, 1), (2, 1), (2, 0), (1, 0), (0, 0)]          # snake layout like the reference image
shade = ["#1f77b4", "#3a5fa8", "#3b4a9a", "#2c3e8f", "#2a7fc0", "#1b8ab8"]
for (t, d), (x, y), c in zip(steps, pos, shade):
    ax.add_patch(FancyBboxPatch((x + .06, y + .12), .84, .74, boxstyle="round,pad=0.02", fc=c, ec="none"))
    ax.text(x + .48, y + .62, t, color="white", ha="center", fontsize=12.5, weight="bold")
    ax.text(x + .48, y + .34, d, color="white", ha="center", fontsize=10, va="center")
for (x1, y1), (x2, y2) in zip(pos[:-1], pos[1:]):
    if y1 == y2:
        dx = .1 if x2 > x1 else -.1
        ax.annotate("", (x2 + .48 - .44 * np.sign(x2 - x1) - dx * 0, y2 + .49), (x1 + .48 + .44 * np.sign(x2 - x1), y1 + .49),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.8))
    else:
        ax.annotate("", (x2 + .48, y2 + .88), (x1 + .48, y1 + .1), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.8))
ax.set_title("Our project as a data-visualisation pipeline", fontsize=15, weight="bold")
plt.tight_layout(); plt.savefig(f"{OUT}/00_story.png", dpi=130); plt.show()
''')

# ---------------------------------------------------------------- step 1
md(r'''
## Step 1: Identify the data sources
We did not download a dataset. We **generated** graphs with our own Python generators (`datagen/generators.py`, `datagen/hub_sweep.py`) so that we control their shape:

* **uniform**: every vertex has about the same number of edges (no hubs)
* **powerlaw_balanced, in_heavy, out_heavy, multi_hub**: a few vertices have far more edges than the rest (skew)
* **hub_sweep**: we plant 1–8 hubs and slowly raise how much of the work goes through them, to find where hub-split starts to win

Each graph is stored as a Parquet edge list (`src, dst`). Here we load the final table (one row per graph) and look at what we have.
''')

code(r'''
df = pd.read_csv(TABLE)
df["vertices"] = np.expm1(df.log_num_vertices).round().astype(int)
df["edges"] = np.expm1(df.log_num_edges).round().astype(int)
df["two_hop_paths"] = df.raw_work_sum                       # exact number of (a)->(b)->(c) results
fam = df.family.value_counts()
dims = pd.DataFrame({"graphs": [len(df)], "total edges": [f"{df.edges.sum()/1e6:.1f} M"],
                     "vertices per graph": [f"{df.vertices.min():,} – {df.vertices.max():,}"],
                     "edges per graph": [f"{df.edges.min()/1e6:.2f} M – {df.edges.max()/1e6:.1f} M"],
                     "two-hop paths per graph": [f"{df.two_hop_paths.min()/1e6:.1f} M – {df.two_hop_paths.max()/1e6:.0f} M"]})
display(dims)

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
fam.sort_values().plot.barh(ax=ax[0], color="#4C78A8")
for i, v in enumerate(fam.sort_values()): ax[0].text(v + .3, i, str(v), va="center")
ax[0].set_title("Graphs per family"); ax[0].set_xlabel("number of graphs")
cmap = dict(zip(fam.index, plt.cm.tab10.colors))
jit = np.random.RandomState(1).uniform(-.25, .25, len(df))          # small sideways shift so dots do not hide each other
for f, g in df.groupby("family"):
    ax[1].scatter(g.edges / 1e6 + jit[g.index], g.two_hop_paths / 1e6, s=g.work_max_share * 600 + 12, alpha=.6, color=cmap[f], edgecolor="k", lw=.4, label=f)
ax[1].set_yscale("log"); ax[1].set_xlabel("edges (millions)"); ax[1].set_ylabel("two-hop paths = work (millions, log scale)")
ax[1].set_title("Same size, very different work (bubble = busiest vertex's share)"); ax[1].legend(fontsize=8, markerscale=.6, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/01_dataset.png", dpi=130); plt.show()
RES["n_graphs"] = len(df); RES["total_edges_M"] = round(df.edges.sum() / 1e6, 1)
small = df[df.edges < 1e6]
display(Markdown(f"""**How we got it:** we counted the rows per family (bar chart). For the bubble chart, each graph is placed by its number of edges (x) and its number of two-hop results, i.e. the work (y); a bigger bubble means more of the work sits on one vertex.
**What it tells us:** {len(df)} graphs and {df.edges.sum()/1e6:.0f} M edges in total. Half the graphs ({fam['hub_sweep']}) are the hub sweep, built to test the hub case.
Graphs with the **same** number of edges (about 0.7 M) need anywhere from **{small.two_hop_paths.min()/1e6:.1f} M to {small.two_hop_paths.max()/1e6:.0f} M** paths of work, and the big bubbles (one heavy vertex) sit at the top.
So **size alone cannot pick the strategy**; we need features that describe the graph's shape."""))
''')

# ---------------------------------------------------------------- step 2
md(r'''
## Step 2: Extract the data (Spark benchmark)
**Big data tool:** Apache Spark 3.5.5 (local mode) + GraphFrames 0.8.3, reading Parquet.
For every graph we started a fresh Spark session, loaded the edges (`spark.read.parquet`, in `bench/phase1_worker.py`),
checked that every strategy returns the **same** number of paths (correctness gate), did a warm-up run and then
several timed runs in shuffled order. We keep the **median** time per graph and strategy.

*This step took hours, so we do not re-run Spark in the review. Its output (the medians) is already in the table.*
Below we look at what the timings say.
''')

code(r'''
T = np.column_stack([df[f"median_total_s__{s}"].values for s in STRATS])        # 62 x 3 median seconds
runs_used = int(sum(df[f"n_valid_runs__{s}"].sum() for s in STRATS))
print(f"timed Spark runs behind these medians: {runs_used}  (62 graphs x 3 strategies, 2-5 runs each)")
sw = df[(df.family == "hub_sweep") & (df.hub_count == 1) & (df.seed == 101)].sort_values("hub_work_share_planned")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for k, s in enumerate(STRATS):
    ax[0].plot(sw.hub_work_share_planned * 100, sw[f"median_total_s__{s}"], "o-", lw=2, color=COL[NAME[s]], label=NAME[s])
ax[0].set_xlabel("share of the work going through the hub (%)"); ax[0].set_ylabel("runtime (s)")
ax[0].set_title("Hub sweep (1 hub): runtime as the hub gets heavier"); ax[0].legend()
avg = pd.Series(T.mean(0), index=[NAME[s] for s in STRATS])
avg.plot.bar(ax=ax[1], color=[COL[n] for n in avg.index], rot=0)
for i, v in enumerate(avg): ax[1].text(i, v + .3, f"{v:.1f}s", ha="center")
ax[1].set_title("Average runtime per strategy (all 62 graphs)"); ax[1].set_ylabel("seconds")
plt.tight_layout(); plt.savefig(f"{OUT}/02_runtimes.png", dpi=130); plt.show()
cross = sw[sw["median_total_s__hypercube_unified__aqe_off"] < sw["median_total_s__edgejoin__aqe_off"]].hub_work_share_planned.min()
display(Markdown(f"""**How we got it:** the median Spark runtime of each strategy on the one-hub graphs (line chart), and the average over all graphs (bar chart).
**What it tells us:** while the hub carries little work, edge-join is fastest. When the hub carries about **{cross*100:.0f}%** or more of the work,
edge-join's time shoots up because one Spark task gets all the hub's paths, and **hub-split** becomes the fastest.
The fastest choice **changes with the graph's shape**, and that is why a model is needed."""))
''')

# ---------------------------------------------------------------- step 3
md(r'''
## Step 3: Map the data (feature engineering + labels)
Each graph is turned into **13 numbers** that can be computed *before* running the query:

* **11 base features** (`ml/build_feature_table.py`): size (vertices, edges, active fraction), degree skew (in/out Gini, top-0.1% degrees) and two-hop work (total paths, busiest vertex's share, number of hub vertices, hubs' share).
* **2 derived features** (computed here): `log_hub_work` = paths through the hubs, and `log_max_work` = paths through the single busiest vertex.

Counts are put through `log(1 + x)` so that huge numbers become comparable. **Label** = the strategy with the lowest median runtime.
''')

code(r'''
df["log_hub_work"] = np.log1p(df.raw_work_sum * df.work_in_hubs_share)
df["log_max_work"] = np.log1p(df.raw_work_sum * df.work_max_share)
FEATS = list(FEATURE_COLUMNS) + ["log_hub_work", "log_max_work"]
X = df[FEATS].values
y = df.best_class.values
LOGT = np.log(T)
print(f"input matrix X: {X.shape[0]} graphs x {X.shape[1]} features")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1.5, 1]})
corr = df[FEATS].corr()
im = ax[0].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_xticks(range(len(FEATS))); ax[0].set_xticklabels(FEATS, rotation=60, ha="right", fontsize=8)
ax[0].set_yticks(range(len(FEATS))); ax[0].set_yticklabels(FEATS, fontsize=8); ax[0].grid(False)
for i in range(len(FEATS)):
    for j in range(len(FEATS)):
        v = corr.iloc[i, j]; ax[0].text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(v) > .6 else "black")
plt.colorbar(im, ax=ax[0], fraction=.04); ax[0].set_title("Feature correlation heatmap")
cnt = pd.Series(y).value_counts().reindex(CLASSES)
bars = ax[1].bar(["edge-join", "hub-split", "native"], cnt.values, color=[COL["edge-join"], COL["hub-split"], COL["native"]])
for b, v in zip(bars, cnt.values): ax[1].text(b.get_x() + b.get_width() / 2, v + .5, f"{v} ({v/len(y):.0%})", ha="center")
ax[1].set_title("Labels: which strategy was fastest"); ax[1].set_ylabel("graphs")
plt.tight_layout(); plt.savefig(f"{OUT}/03_features_labels.png", dpi=130); plt.show()
MAJ = cnt.max() / cnt.sum(); RES["class_counts"] = cnt.to_dict()
display(Markdown(f"""**How we got it:** the Pearson correlation between every pair of the 13 features (heatmap; red = they rise together, blue = one rises as the other falls), and a count of the labels (bar chart).
**What it tells us:** the features form three groups that move together: **size** (vertices, edges), **degree skew** (Gini and top-0.1% degrees, which are high when active_fraction is low) and **hub work** (the work and hub features at the bottom right).
The groups are only weakly related to each other, so each one adds new information and all of them are kept.
The labels are **imbalanced**: edge-join wins {cnt['edgejoin']} of {len(y)} graphs ({MAJ:.1%}). So a "model" that always says edge-join already gets {MAJ:.1%} accuracy.
That is our **baseline**, and it is why we also report **balanced accuracy** and **macro-F1**, which do not reward ignoring the rare classes."""))
''')

# ---------------------------------------------------------------- step 4
md(r'''
## Step 4: Upload into one standard table
All the pieces (graph features from Step 1/3 and Spark medians from Step 2) are joined into **one table**, `datasets/derived/training/combined_table.csv`,
by `ml/build_combined_table.py`. One row = one graph. This is the only file the models need.
''')

code(r'''
print("combined table:", df.shape[0], "rows x", pd.read_csv(TABLE).shape[1], "columns")
raw_gb = 0.68                                       # Parquet edge files of the 62 graphs on disk
table_kb = os.path.getsize(TABLE) / 1024
print(f"size before preprocessing: about {raw_gb} GB of edges  ->  after: {table_kb:.0f} KB table  (~{raw_gb*1024*1024/table_kb:,.0f}x smaller)")
display(df[["graph_id", "family", "vertices", "edges", "work_max_share", "work_in_hubs_share", "best_class"]].head(6))
''')

# ---------------------------------------------------------------- step 5
md(r'''
## Step 5: Graphically represent: train, validate, test, compare

### 5a. Which model? Classifier comparison (leave-one-graph-out)
**Leave-one-graph-out (LOGO):** train on 61 graphs and test on the 1 left out; repeat 62 times so every graph is tested once, on a model that never saw it.

Models compared:
* **Majority baseline**: always predicts edge-join (the most common label)
* **Rule baseline**: a hand-written rule, "if the graph has hubs use hub-split, else native"
* **Logistic regression**, **Random Forest** (300 trees), **Extra-Trees** (300 trees): our ML models

**Primary metrics** (higher is better):
* **Accuracy**: share of graphs where the predicted strategy is right
* **Balanced accuracy**: the average of the per-class hit rates (a model that ignores hub-split and native scores only 33%)
* **Macro-F1**: the average F1 over the three classes (it punishes both missed and false picks)
''')

code(r'''
def make_models():
    return {
        "Logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=NTREES, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        "Extra-Trees": lambda: ExtraTreesClassifier(n_estimators=NTREES, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
    }

def rule_pred(d):
    return np.where(d.work_in_hubs_share > 0, "hubsplit", "native")

def scores(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                macro_f1=f1_score(yt, yp, average="macro", labels=CLASSES))

def leave_one_out(idx):
    # leave-one-graph-out over the rows in idx: returns predictions for each of them
    preds = {m: np.empty(len(idx), dtype=object) for m in ["Majority baseline", "Rule baseline", *make_models()]}
    for k, i in enumerate(idx):
        tr = np.delete(idx, k)
        vals, counts = np.unique(y[tr], return_counts=True)
        preds["Majority baseline"][k] = vals[counts.argmax()]
        preds["Rule baseline"][k] = rule_pred(df.iloc[[i]])[0]
        for name, mk in make_models().items():
            preds[name][k] = mk().fit(X[tr], y[tr]).predict(X[[i]])[0]
    return preds

ALL = np.arange(len(df))
LOGO_PRED = leave_one_out(ALL)
clf = pd.DataFrame({m: scores(y, p) for m, p in LOGO_PRED.items()}).T
display(clf.round(3))
RES["classifier_LOGO"] = clf.round(4).to_dict(orient="index")

fig, ax = plt.subplots(figsize=(12, 4.4)); xs = np.arange(len(clf)); w = .26
for j, (m, lab, c) in enumerate([("accuracy", "Accuracy", "#4C78A8"), ("balanced_accuracy", "Balanced accuracy", "#F58518"), ("macro_f1", "Macro-F1", "#54A24B")]):
    b = ax.bar(xs + (j - 1) * w, clf[m], w, label=lab, color=c)
    ax.bar_label(b, fmt="%.2f", fontsize=7.5, padding=1)
ax.axhline(clf.loc["Majority baseline", "accuracy"], ls="--", c="k", lw=1, label="majority accuracy (80.6%)")
ax.set_xticks(xs); ax.set_xticklabels(clf.index); ax.set_ylim(0, 1.1); ax.set_ylabel("score")
ax.set_title("Classifier comparison on held-out graphs (leave-one-graph-out, 62 tests)"); ax.legend(fontsize=8, ncol=4, loc="upper left")
plt.tight_layout(); plt.savefig(f"{OUT}/05a_classifiers.png", dpi=130); plt.show()
best = clf.drop(["Majority baseline", "Rule baseline"]).accuracy.idxmax()
display(Markdown(f"""**How we got it:** each model predicted every graph while that graph was hidden from training (62 rounds), and we scored the 62 predictions.
**What it tells us:** the majority baseline has high accuracy ({clf.loc['Majority baseline','accuracy']:.1%}) but only **{clf.loc['Majority baseline','balanced_accuracy']:.1%} balanced accuracy**, because it never finds hub-split or native.
**{best}** is best on accuracy (**{clf.loc[best,'accuracy']:.1%}**) and macro-F1 (**{clf.loc[best,'macro_f1']:.2f}** vs {clf.loc['Majority baseline','macro_f1']:.2f}).
Logistic regression has the highest balanced accuracy (**{clf.loc['Logistic regression','balanced_accuracy']:.1%}**) but lower accuracy, so it over-predicts the rare classes.
The hand-written rule is far worse than every learned model: a fixed rule does not work, and learning from data does."""))
''')

md(r'''
### 5b. Train vs validation vs test
To report the usual three numbers we split the graphs:
* **Test set** = the 9 hub-sweep graphs generated with seed 202. They are kept aside and never used for training or tuning.
* **Training set** = the other 53 graphs.
* **Train score** = fit on the 53 and score on the same 53 (shows how well the model memorises).
* **Validation score** = leave-one-graph-out *inside* the 53 (an honest estimate while building).
* **Test score** = fit on the 53 and score on the 9 unseen graphs (the final exam).
''')

code(r'''
test_idx = np.where((df.family == "hub_sweep") & (df.seed == 202))[0]
train_idx = np.setdiff1d(ALL, test_idx)
print(f"train = {len(train_idx)} graphs, test = {len(test_idx)} graphs (unseen)")
VAL_PRED = leave_one_out(train_idx)
rows = []
for m in ["Majority baseline", "Logistic regression", "Random Forest", "Extra-Trees"]:
    if m == "Majority baseline":
        vals, counts = np.unique(y[train_idx], return_counts=True); maj = vals[counts.argmax()]
        p_tr, p_te = np.full(len(train_idx), maj, dtype=object), np.full(len(test_idx), maj, dtype=object)
    else:
        mdl = make_models()[m]().fit(X[train_idx], y[train_idx]); p_tr, p_te = mdl.predict(X[train_idx]), mdl.predict(X[test_idx])
    for split, yt, yp in [("Train", y[train_idx], p_tr), ("Validation", y[train_idx], VAL_PRED[m]), ("Test", y[test_idx], p_te)]:
        rows.append(dict(model=m, split=split, **scores(yt, yp)))
tvt = pd.DataFrame(rows)
display(tvt.pivot(index="model", columns="split", values=["accuracy", "balanced_accuracy"]).round(3))
RES["train_val_test"] = tvt.round(4).to_dict(orient="records")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
mods = ["Majority baseline", "Logistic regression", "Random Forest", "Extra-Trees"]; mc = ["#B0B0B0", "#F58518", "#4C78A8", "#2E8B57"]
for a, met, title in [(ax[0], "accuracy", "Accuracy"), (ax[1], "balanced_accuracy", "Balanced accuracy")]:
    for j, (m, c) in enumerate(zip(mods, mc)):
        v = tvt[tvt.model == m].set_index("split").loc[["Train", "Validation", "Test"], met]
        b = a.bar(np.arange(3) + (j - 1.5) * .2, v, .2, color=c, label=m); a.bar_label(b, fmt="%.2f", fontsize=7)
    a.set_xticks(range(3)); a.set_xticklabels(["Train (53)", "Validation (53, LOGO)", "Test (9 unseen)"]); a.set_title(title); a.set_ylim(0, 1.12)
ax[0].set_ylabel("score"); fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=4, fontsize=9, bbox_to_anchor=(.5, 1.06))
plt.tight_layout(); plt.savefig(f"{OUT}/05b_train_val_test.png", dpi=130, bbox_inches="tight"); plt.show()
g = tvt.set_index(["model", "split"])
te = tvt[(tvt.split == "Test") & (tvt.model != "Majority baseline")].set_index("model")
bt = te.accuracy.idxmax()
display(Markdown(f"""**How we got it:** the same models are trained on 53 graphs. We scored them on those 53 (train), with leave-one-out inside the 53 (validation), and on the 9 graphs they never saw (test).
**What it tells us:** the tree models reach 100% on train because they memorise the data, so the train score is not trusted; validation and test are the honest numbers.
On validation, the tree models reach **{g.loc[('Extra-Trees','Validation'),'accuracy']:.1%}** accuracy versus {g.loc[('Majority baseline','Validation'),'accuracy']:.1%} for the baseline.
On the 9 unseen test graphs **every ML model beats the baseline**: the best is **{bt}** with **{te.loc[bt,'accuracy']:.1%}** accuracy and **{te.loc[bt,'balanced_accuracy']:.1%}** balanced accuracy,
against **{g.loc[('Majority baseline','Test'),'accuracy']:.1%}** and **{g.loc[('Majority baseline','Test'),'balanced_accuracy']:.1%}** for the baseline.
The test set is small (9 graphs), so one graph changes accuracy by 11 points. This is why our main numbers use leave-one-graph-out over all 62."""))
''')

md(r'''
### 5c. Where does the best classifier make mistakes? (confusion matrix) and what does it look at? (feature importance)
''')

code(r'''
cm = confusion_matrix(y, LOGO_PRED["Extra-Trees"], labels=CLASSES)
et_full = ExtraTreesClassifier(n_estimators=NTREES, class_weight="balanced_subsample", random_state=SEED, n_jobs=1).fit(X, y)
imp = pd.Series(et_full.feature_importances_, index=FEATS).sort_values()
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.3]})
ax[0].imshow(cm, cmap="Blues"); ax[0].grid(False)
lab = ["edge-join", "hub-split", "native"]
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(lab); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(lab)
for i in range(3):
    for j in range(3): ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("actual fastest"); ax[0].set_title("Extra-Trees confusion matrix (LOGO)")
imp.plot.barh(ax=ax[1], color="#2E8B57"); ax[1].set_title("Which features the model uses most"); ax[1].set_xlabel("importance")
plt.tight_layout(); plt.savefig(f"{OUT}/05c_confusion_importance.png", dpi=130); plt.show()
rec = cm.diagonal() / cm.sum(1)
display(Markdown(f"""**How we got it:** the confusion matrix counts actual vs predicted for the 62 held-out predictions (the diagonal is correct). The importance bars show how much each feature reduced errors inside the trees.
**What it tells us:** edge-join is found **{cm[0,0]} of {cm[0].sum()}** times, hub-split **{cm[1,1]} of {cm[1].sum()}** and native **{cm[2,2]} of {cm[2].sum()}**. Most mistakes ({cm[0,2] + cm[2,0]} of {cm.sum() - np.trace(cm)}) are between edge-join and native,
whose runtimes are often close, and the native class has only 6 examples to learn from. The most-used features are **{imp.index[-1]}**, **{imp.index[-2]}** and **{imp.index[-3]}**:
the model decides mostly by how much work piles onto the busiest vertices, which matches what we saw in Step 2."""))
''')

md(r'''
### 5d. The selector: predict each strategy's runtime, pick the smallest
A classifier only says *which* strategy wins. Our **ML selector** goes one step further: a Random Forest **regressor** predicts the runtime of **all three** strategies for a graph, and we pick the smallest.
This is what we would actually deploy. It is evaluated with leave-one-graph-out too.
''')

code(r'''
def fit_reg(Xtr, Ytr):
    return RandomForestRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0, random_state=SEED, n_jobs=1).fit(Xtr, Ytr)

PRED = np.full_like(LOGT, np.nan)
for i in ALL:
    tr = np.delete(ALL, i); PRED[i] = fit_reg(X[tr], LOGT[tr]).predict(X[[i]])[0]
pick = PRED.argmin(1)
err = np.median(np.abs(np.exp(PRED) / T - 1)) * 100
fig, ax = plt.subplots(figsize=(6, 5))
for k, s in enumerate(STRATS): ax.scatter(T[:, k], np.exp(PRED[:, k]), s=22, alpha=.75, color=COL[NAME[s]], label=NAME[s])
lim = [T.min() * .8, T.max() * 1.2]; ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("measured runtime (s)"); ax.set_ylabel("predicted runtime (s)")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(matplotlib.ticker.FixedLocator([5, 10, 20, 50, 100, 200])); a.set_major_formatter(matplotlib.ticker.ScalarFormatter()); a.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax.set_title("Selector's runtime prediction (graph unseen)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/05d_regressor.png", dpi=130); plt.show()
RES["regressor_median_error_pct"] = round(float(err), 2)
display(Markdown(f"""**How we got it:** for each held-out graph the regressor predicted three runtimes. Each dot is one prediction against the real Spark time (both axes are on a log scale, so 1, 10 and 100 s are evenly spaced).
**What it tells us:** the dots sit close to the dashed "perfect" line. The median prediction error is **{err:.1f}%**, which is accurate enough to rank the three strategies correctly for most graphs."""))
''')

md(r'''
### 5e. Is our selector actually better? Comparison with the baselines
We compare the ML selector with what a Spark user would do without it:
* **Always native**: the default GraphFrames way
* **Always edge-join**: the single best fixed choice
* **Always hub-split**
* **Rule**: hubs → hub-split, else native

**Primary metrics for the selector:**
* **Total runtime**: the sum of the real Spark time of the chosen strategy over all 62 graphs (lower is better)
* **Mean regret**: on average, how much slower the chosen strategy is than the fastest of the three for that graph (lower is better; 0% = always picked the fastest)
''')

code(r'''
def policy_scores(rows_idx, choice):
    t = T[rows_idx, choice]; best_t = T[rows_idx].min(1); nat = T[rows_idx, STRATS.index("native__aqe_on")]
    return dict(total_time_s=float(t.sum()), mean_regret_pct=float(np.mean(t / best_t - 1) * 100),
                picked_fastest_pct=float(np.mean(choice == T[rows_idx].argmin(1)) * 100), speedup_vs_native=float(np.exp(np.mean(np.log(nat / t)))))

def policies(rows_idx, pred):
    n = len(rows_idx)
    rule = np.where(df.work_in_hubs_share.values[rows_idx] > 0, STRATS.index("hypercube_unified__aqe_off"), STRATS.index("native__aqe_on"))
    return {"Always native": np.full(n, 1), "Always edge-join": np.full(n, 0), "Always hub-split": np.full(n, 2),
            "Rule": rule, "ML selector": pred[rows_idx].argmin(1)}

POL = pd.DataFrame({k: policy_scores(ALL, v) for k, v in policies(ALL, PRED).items()}).T
display(POL.round(2)); RES["policies_LOGO"] = POL.round(4).to_dict(orient="index")
order = ["Always native", "Always hub-split", "Rule", "Always edge-join", "ML selector"]
cols = [COL["ML selector"] if o == "ML selector" else COL["baseline"] for o in order]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
b = ax[0].barh(order, POL.loc[order, "total_time_s"], color=cols); ax[0].bar_label(b, fmt="%.0f s", padding=3)
ax[0].set_title("Total runtime over 62 graphs (lower is better)"); ax[0].set_xlabel("seconds"); ax[0].set_xlim(0, POL.total_time_s.max() * 1.18)
b = ax[1].barh(order, POL.loc[order, "mean_regret_pct"], color=cols); ax[1].bar_label(b, fmt="%.1f%%", padding=3)
ax[1].set_title("Mean regret: % slower than the best strategy (lower is better)"); ax[1].set_xlabel("%"); ax[1].set_xlim(0, POL.mean_regret_pct.max() * 1.18)
plt.tight_layout(); plt.savefig(f"{OUT}/05e_selector_vs_baselines.png", dpi=130); plt.show()
p = POL.loc
display(Markdown(f"""**How we got it:** for every graph each policy chooses one strategy. We add up the real measured Spark time of that choice (left) and average how far it is from the fastest choice (right).
**What it tells us:** the ML selector has the **lowest total time ({p['ML selector','total_time_s']:.0f} s)** and the **lowest regret ({p['ML selector','mean_regret_pct']:.1f}%)**.
That is **{(1-p['ML selector','total_time_s']/p['Always native','total_time_s'])*100:.0f}% less time than always using native** (the default) and
**{(1-p['ML selector','total_time_s']/p['Always edge-join','total_time_s'])*100:.0f}% less than the best fixed strategy, edge-join**, whose regret ({p['Always edge-join','mean_regret_pct']:.1f}%) is almost double ours.
It picks the fastest strategy for **{p['ML selector','picked_fastest_pct']:.0f}%** of the graphs."""))
''')

md(r'''
### 5f. Same comparison on the 9 unseen test graphs
Here the selector is trained on the 53 training graphs only and judged on the 9 test graphs from Step 5b.
''')

code(r'''
reg53 = fit_reg(X[train_idx], LOGT[train_idx])
PRED_T = np.full_like(LOGT, np.nan); PRED_T[test_idx] = reg53.predict(X[test_idx])
POL_T = pd.DataFrame({k: policy_scores(test_idx, v) for k, v in policies(test_idx, PRED_T).items()}).T
display(POL_T.round(2)); RES["policies_test9"] = POL_T.round(4).to_dict(orient="index")
fig, ax = plt.subplots(figsize=(8, 3.8))
o2 = POL_T.loc[order, "mean_regret_pct"]
b = ax.bar(o2.index, o2.values, color=cols); ax.bar_label(b, fmt="%.1f%%")
ax.set_ylabel("mean regret (%)"); ax.set_title("Unseen test graphs: % slower than the best strategy (lower is better)")
plt.tight_layout(); plt.savefig(f"{OUT}/05f_test_policies.png", dpi=130); plt.show()
display(Markdown(f"""**How we got it:** the same regret measure as in 5e, but only on the 9 held-out hub-sweep graphs, with a selector that never saw them.
**What it tells us:** on new graphs the selector is still the best: **{POL_T.loc['ML selector','mean_regret_pct']:.1f}%** regret against {POL_T.loc['Always edge-join','mean_regret_pct']:.1f}% for always edge-join
and {POL_T.loc['Always native','mean_regret_pct']:.1f}% for always native. These test graphs include heavy-hub cases, where a fixed choice is costly and the selector helps the most."""))
''')

md(r'''
### 5g. How the time savings add up, and the headline numbers
''')

code(r'''
ordr = np.argsort(T[:, 1])                                   # graphs from fastest to slowest (under native)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.6, 1]})
xs = np.arange(1, len(df) + 1)
cum = {n: np.cumsum(T[ordr, k]) for n, k in [("Always native", 1), ("Always edge-join", 0)]}
cum["ML selector"] = np.cumsum(T[ordr, pick[ordr]])
ax[0].fill_between(xs, cum["Always native"], color=COL["native"], alpha=.35, label=f"Always native ({cum['Always native'][-1]:.0f} s)")
ax[0].fill_between(xs, cum["Always edge-join"], color=COL["edge-join"], alpha=.45, label=f"Always edge-join ({cum['Always edge-join'][-1]:.0f} s)")
ax[0].fill_between(xs, cum["ML selector"], color=COL["ML selector"], alpha=.75, label=f"ML selector ({cum['ML selector'][-1]:.0f} s)")
ax[0].set_xlabel("number of graphs run (quickest to slowest)"); ax[0].set_ylabel("total seconds so far")
ax[0].set_title("Cumulative runtime: the gap grows on big, skewed graphs"); ax[0].legend(loc="upper left")
# gauge: % of time saved compared with the default (always native)
saved = (1 - POL.loc["ML selector", "total_time_s"] / POL.loc["Always native", "total_time_s"]) * 100
ax[1].axis("off"); ax[1].set_xlim(-1.2, 1.2); ax[1].set_ylim(-.35, 1.15); ax[1].set_aspect("equal")
ax[1].add_patch(Wedge((0, 0), 1, 0, 180, width=.3, fc="#e6e6e6"))
ax[1].add_patch(Wedge((0, 0), 1, 180 - 180 * saved / 100, 180, width=.3, fc=COL["ML selector"]))
ax[1].text(0, .15, f"{saved:.0f}%", ha="center", fontsize=30, weight="bold"); ax[1].text(0, -.15, "less total time than default (native)", ha="center", fontsize=10)
ax[1].text(-1, -.08, "0%", ha="center"); ax[1].text(1, -.08, "100%", ha="center"); ax[1].set_title("Time saved by the ML selector")
plt.tight_layout(); plt.savefig(f"{OUT}/05g_cumulative_gauge.png", dpi=130); plt.show()
RES["time_saved_vs_native_pct"] = round(float(saved), 2)
last5 = (cum["Always edge-join"][-1] - cum["Always edge-join"][-6]) - (cum["ML selector"][-1] - cum["ML selector"][-6])
display(Markdown(f"""**How we got it:** graphs are sorted from quickest to slowest (by native runtime), and each area shows the running total of time if we use that policy (area chart). The gauge shows the final saving against the default.
**What it tells us:** against **native** the gap opens steadily, graph after graph. Against **edge-join** the selector is about equal on most graphs (both pick edge-join),
and then pulls ahead on the last, slowest, hub-heavy graphs, where it switches to hub-split: **{last5:.0f} s** saved on the last 5 graphs alone.
Overall the selector saves **{saved:.0f}%** of the time of the default GraphFrames strategy."""))
''')

# ---------------------------------------------------------------- step 6
md(r'''
## Step 6: Save and share reports
We save three things:
1. `datasets/derived/review/results.json`: every number shown above
2. `datasets/derived/review/*.png`: all the charts (used in our slides and report)
3. **Tableau "Decision Explorer"** (`datasets/derived/review/tableau/`). These are not the same charts again. They are a **per-graph decision log** that Tableau can filter and drill into:
   for each graph, what the selector predicted, what it picked, what was actually fastest, and how many seconds it saved compared with native and edge-join.
   In Tableau you can filter by family or hub share and see **where** the selector helps or makes mistakes. That is something the static charts above cannot do.
''')

code(r'''
TAB = os.path.join(OUT, "tableau"); os.makedirs(TAB, exist_ok=True)
dec = pd.DataFrame({
    "graph_id": df.graph_id, "family": df.family, "hub_count": df.hub_count.fillna(0).astype(int),
    "hub_work_share_pct": (df.work_in_hubs_share * 100).round(1), "busiest_vertex_share_pct": (df.work_max_share * 100).round(1),
    "vertices": df.vertices, "edges": df.edges, "two_hop_paths": df.two_hop_paths,
    **{f"pred_s_{NAME[s].replace('-', '_')}": np.exp(PRED[:, k]).round(2) for k, s in enumerate(STRATS)},
    **{f"actual_s_{NAME[s].replace('-', '_')}": T[:, k].round(2) for k, s in enumerate(STRATS)},
    "selector_pick": [NAME[STRATS[k]] for k in pick], "actual_fastest": [NAME[STRATS[k]] for k in T.argmin(1)],
})
dec["correct"] = dec.selector_pick == dec.actual_fastest
dec["selector_s"] = T[ALL, pick].round(2)
dec["saved_vs_native_s"] = (T[:, 1] - dec.selector_s).round(2)
dec["saved_vs_edgejoin_s"] = (T[:, 0] - dec.selector_s).round(2)
dec["regret_pct"] = ((dec.selector_s / T.min(1) - 1) * 100).round(1)
kpi = pd.DataFrame([dict(policy=k, **v) for k, v in RES["policies_LOGO"].items()])
dec.to_csv(f"{TAB}/decision_explorer.csv", index=False); kpi.to_csv(f"{TAB}/policy_kpis.csv", index=False)
json.dump(RES, open(f"{OUT}/results.json", "w"), indent=1, default=float)
print("wrote", f"{TAB}/decision_explorer.csv", dec.shape, "|", f"{TAB}/policy_kpis.csv", kpi.shape, "| results.json")

# Preview of the Tableau view, read back from the file exactly as Tableau would
L = pd.read_csv(f"{TAB}/decision_explorer.csv")
byfam = L.groupby("family")[["saved_vs_native_s", "saved_vs_edgejoin_s"]].sum().sort_values("saved_vs_native_s")
fig, ax = plt.subplots(figsize=(10, 4))
yy = np.arange(len(byfam))
ax.barh(yy - .2, byfam.saved_vs_native_s, .4, color=COL["native"], label="seconds saved vs always native")
ax.barh(yy + .2, byfam.saved_vs_edgejoin_s, .4, color=COL["edge-join"], label="seconds saved vs always edge-join")
ax.set_yticks(yy); ax.set_yticklabels(byfam.index); ax.axvline(0, c="k", lw=.8); ax.set_xlabel("seconds saved by the ML selector (negative = lost)")
for yv, (a_, b_) in enumerate(zip(byfam.saved_vs_native_s, byfam.saved_vs_edgejoin_s)):
    ax.text(max(a_, 0) + 4, yv - .2, f"{a_:.0f}s", va="center", fontsize=8); ax.text(max(b_, 0) + 4, yv + .2, f"{b_:.0f}s", va="center", fontsize=8)
ax.set_title("Tableau preview: where does the selector save time?"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(f"{OUT}/06_tableau_preview.png", dpi=130); plt.show()
wrong = L[~L.correct].sort_values("regret_pct", ascending=False)
print(f"selector picked the fastest strategy on {int(L.correct.sum())} of {len(L)} graphs; the misses:")
display(wrong[["graph_id", "family", "selector_pick", "actual_fastest", "regret_pct", "saved_vs_native_s"]].head(11))
best_f = byfam.saved_vs_native_s.idxmax()
display(Markdown(f"""**How we got it:** we summed the per-graph "seconds saved" columns of the Tableau file by family, and listed the graphs where the pick was not the fastest.
**What it tells us:** most of the saving comes from **{best_f}** graphs (hub-heavy ones), which is exactly where a smart choice matters. Against edge-join the saving is smaller,
because edge-join is already right for most graphs. The selector is wrong on {len(wrong)} graphs, but these are mostly small, close calls: together they cost only
**{(wrong.selector_s - L.loc[wrong.index, ['actual_s_edge_join','actual_s_native','actual_s_hub_split']].min(axis=1)).sum():.0f} s** more than the fastest choice,
against **{byfam.saved_vs_native_s.sum():.0f} s** saved compared with native over all graphs.
**In Tableau:** *Connect → Text file → `decision_explorer.csv`*, put `family` on Rows, `SUM(saved_vs_native_s)` on Columns, `correct` on Colour, and add `hub_work_share_pct` as a filter."""))
''')

# ---------------------------------------------------------------- reproducibility
md(r'''
## Reproducibility check
Fixed seeds, `n_jobs=1` and no random sampling mean that every run gives exactly the same numbers. The cell below compares this run with the numbers in our report and slides.
''')

code(r'''
expected = {   # numbers in our report / slides (leave-one-graph-out, 62 graphs)
    "majority accuracy": (clf.loc["Majority baseline", "accuracy"], 0.806),
    "majority balanced accuracy": (clf.loc["Majority baseline", "balanced_accuracy"], 0.333),
    "logistic regression balanced accuracy": (clf.loc["Logistic regression", "balanced_accuracy"], 0.753),
    "random forest accuracy": (clf.loc["Random Forest", "accuracy"], 0.839),
    "extra-trees accuracy": (clf.loc["Extra-Trees", "accuracy"], 0.855),
    "extra-trees macro-F1": (clf.loc["Extra-Trees", "macro_f1"], 0.703),
    "selector total time (s)": (POL.loc["ML selector", "total_time_s"], 1070),
    "always edge-join total time (s)": (POL.loc["Always edge-join", "total_time_s"], 1226),
    "always native total time (s)": (POL.loc["Always native", "total_time_s"], 1810),
    "selector mean regret (%)": (POL.loc["ML selector", "mean_regret_pct"], 4.6),
    "always edge-join mean regret (%)": (POL.loc["Always edge-join", "mean_regret_pct"], 8.3),
}
chk = pd.DataFrame([dict(metric=k, this_run=round(v, 3), report=r, match=abs(round(v, 3 if r < 1 else 1 if r < 100 else 0) - r) < 1e-9 or abs(v - r) < (5e-4 if r < 1 else .05 if r < 100 else .5)) for k, (v, r) in expected.items()])
display(chk)
print("ALL NUMBERS MATCH THE REPORT" if chk.match.all() else "SOME NUMBERS DIFFER - check library versions / data file")
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, os.path.join(HERE, "selector.ipynb"))
print("wrote notebooks/selector.ipynb with", len(cells), "cells")
