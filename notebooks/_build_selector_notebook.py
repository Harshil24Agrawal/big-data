#!/usr/bin/env python3
"""Generates notebooks/selector.ipynb (run: python notebooks/_build_selector_notebook.py).
Execute with: jupyter nbconvert --to notebook --execute --inplace notebooks/selector.ipynb  (see docs/PROJECT_BRIEFING.md)."""
import os
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# ML-guided selection of graph-query execution strategies in Spark — training notebook

**Query:** directed two-hop motif `(a)-[e1]->(b); (b)-[e2]->(c)` on Spark 3.5.5 / GraphFrames 0.8.3 (2 vCPU, one machine, local[*]).
**Goal:** from graph-shape features known *before* running, pick the fastest strategy.
**Honesty rules used throughout:** every score is printed next to the *majority-class baseline*; all evaluation is on **held-out graphs**
(leave-one-graph-out, leave-one-structural-group-out, and a held-out generator seed); runtimes are targets only, never inputs; outcomes are
table look-ups of measured medians (no new executions); the model is never tuned on the held-out folds.

| Section | What it does |
|---|---|
| 1 | Load the combined table (31 original graphs + hub-sweep graphs), define features / classes |
| 2 | Look at the data: label balance, runtime vs hub share |
| 3 | **Classifier** — predicts the best *class* (edge-join / hub-splitting / native) |
| 4 | **Runtime regressor** — predicts log runtime of each strategy |
| 5 | **Selector** — picks argmin of predicted runtime, with a safe native fallback |
| 6 | Comparison with baselines and the oracle (accuracy, regret, speed-up) |
| 7 | Save results |
""")

code("""
import os, sys, json, time, math, warnings
import numpy as np, pandas as pd
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, r2_score
warnings.filterwarnings("ignore")

def find_repo(start):
    d = os.path.abspath(start)
    while True:
        if os.path.exists(os.path.join(d, "datasets", "derived", "training", "combined_table.csv")):
            return d
        if os.path.dirname(d) == d:
            raise SystemExit("run this notebook inside the repository (datasets/derived/training/combined_table.csv not found)")
        d = os.path.dirname(d)
REPO = find_repo(os.getcwd())
sys.path.insert(0, REPO)
from ml.build_feature_table import FEATURE_COLUMNS
from ml.build_combined_table import STRATS, CLASS_OF

TABLE = os.path.join(REPO, "datasets", "derived", "training", "combined_table.csv")
OUT = os.path.join(REPO, "datasets", "derived", "selector_v2")
os.makedirs(OUT, exist_ok=True)
NTREES = int(os.environ.get("NB_TREES", "300"))      # 300 for the final run; small for quick tests
SEED = 0
np.random.seed(SEED)
import sklearn, scipy, platform
print("python", platform.python_version(), "| numpy", np.__version__, "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__, "| scipy", scipy.__version__, "| matplotlib", matplotlib.__version__, "| trees", NTREES)
NATIVE = "native__aqe_on"
CLASSES = ["edgejoin", "hubsplit", "native"]
RES = {}                                              # everything reported is stored here and written to results.json
""")

md("## 1. Data\nOne row per graph. Inputs = pre-execution graph features; targets = measured median `total_s` (profiling + query) of 3 strategies.")
code("""
df = pd.read_csv(TABLE)
df["group"] = np.where(df.family == "hub_sweep", "hub_sweep_H" + df.hub_count.fillna(0).astype(int).astype(str), df.family)
# two derived inputs (both computed from the pre-execution features only): absolute work sitting in hubs (hub strategies pay a ~fixed profiling cost, so absolute hub work matters).
df["log_hub_work"] = np.log1p(df.raw_work_sum * df.work_in_hubs_share)
BASE = list(FEATURE_COLUMNS)
# and the work of the single busiest vertex: that vertex's join task is the straggler, so it sets the edge-join runtime floor.
df["log_max_work"] = np.log1p(df.raw_work_sum * df.work_max_share)
FEATS = BASE + ["log_hub_work", "log_max_work"]
T = np.column_stack([df[f"median_total_s__{s}"].values for s in STRATS])      # measured medians, seconds
LOGT = np.log(T)
y_class = df.best_class.values
y_strat = df.best_strategy.values
print(f"{len(df)} graphs x {len(FEATS)} input features; strategies: {STRATS}")
print(df.protocol.value_counts().to_string()); print(); print(df.group.value_counts().to_string())
RES["n_graphs"] = int(len(df)); RES["n_features"] = len(FEATS); RES["features"] = FEATS
RES["protocols"] = df.protocol.value_counts().to_dict()
display(df[["graph_id", "group", "protocol", "work_max_share", "work_in_hubs_share", "raw_work_sum", "best_strategy", "label_margin_pct", "label_clear"]].head(8))
""")

md("""
**Classes.** `edgejoin` = plain edge join (AQE off). `hubsplit` = the hub-aware unified hypercube (splits heavy-hub work; salting was dropped from the sweeps because it was never fastest, see `median_total_s__salt1d__aqe_off` column).
`native` = GraphFrames `find` with AQE on. The class is the strategy family a user would actually choose between.
The label is the strategy with the lowest *median total time*; `label_clear` marks labels whose lead over the runner-up exceeds both runs' IQR.
""")
code("""
cnt = df.best_class.value_counts().reindex(CLASSES).fillna(0).astype(int)
maj_class = cnt.idxmax(); maj_frac = cnt.max() / cnt.sum()
print("class counts:", cnt.to_dict(), f"-> majority class '{maj_class}' = {maj_frac:.1%} of graphs")
print("clear labels:", int(df.label_clear.sum()), "of", len(df))
print(pd.crosstab(df.protocol, df.best_class))
RES["class_counts"] = cnt.to_dict(); RES["majority_class"] = maj_class; RES["majority_frac_global"] = float(maj_frac)
RES["n_clear_labels"] = int(df.label_clear.sum())
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
pd.crosstab(df.group, df.best_class).reindex(columns=CLASSES, fill_value=0).plot.barh(stacked=True, ax=ax[0], color=["#4C78A8", "#E45756", "#72B7B2"])
ax[0].set_title("Best class per structural group"); ax[0].set_xlabel("graphs")
cnt.plot.bar(ax=ax[1], color=["#4C78A8", "#E45756", "#72B7B2"]); ax[1].set_title("Class balance (label = lowest median time)"); ax[1].set_ylabel("graphs")
plt.tight_layout(); plt.savefig(f"{OUT}/label_balance.png", dpi=130); plt.show()
""")

md("### 2. Runtime versus hub share — where (if anywhere) hub-splitting wins\nEach point is one hub-sweep graph (median total time); x = share of two-hop work held by hubs.")
code("""
sw = df[df.family == "hub_sweep"].copy()
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = {"edgejoin__aqe_off": "#4C78A8", "native__aqe_on": "#72B7B2", "hypercube_unified__aqe_off": "#E45756"}
if len(sw):
    for s in STRATS:
        sub = sw.sort_values("work_in_hubs_share")
        axes[0].plot(sub.work_in_hubs_share, sub[f"median_total_s__{s}"], "o", label=s, color=colors[s], alpha=.75)
        axes[1].plot(sub.work_in_hubs_share, sub[f"median_query_s__{s}"], "o", label=s, color=colors[s], alpha=.75)
axes[0].set_title("total time (profiling + query)"); axes[1].set_title("query phase only (excludes profiling)")
for a in axes: a.set_xlabel("hub share of two-hop work"); a.set_ylabel("seconds"); a.grid(alpha=.3)
axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/runtime_vs_hub_share.png", dpi=130); plt.show()
if len(sw):
    g = sw.assign(win=sw.best_strategy).groupby(["hub_work_share_planned"]).win.value_counts().unstack(fill_value=0)
    display(g)
""")

md("""
### 2b. The crossover: hub-splitting vs edge-join against the work through the busiest vertex
The edge join's slowest task is the one holding the busiest vertex, so the work through that single vertex is the natural x-axis. y < 1 means the hub-aware hypercube is faster than the plain edge join (total time, profiling included).
""")
code("""
ratio = df["median_total_s__hypercube_unified__aqe_off"] / df["median_total_s__edgejoin__aqe_off"]
maxw = df.raw_work_sum * df.work_max_share
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for fam, mk in [("hub_sweep", "o"), ("other", "s")]:
    sel = (df.family == "hub_sweep") if fam == "hub_sweep" else (df.family != "hub_sweep")
    ax.scatter(maxw[sel], ratio[sel], marker=mk, s=28, alpha=.75, label="hub-sweep graphs" if fam == "hub_sweep" else "original 31 graphs",
               color="#E45756" if fam == "hub_sweep" else "#4C78A8")
ax.axhline(1, color="k", lw=.8, ls="--"); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("two-hop paths through the single busiest vertex"); ax.set_ylabel("hypercube time / edge-join time")
ax.set_title("Hub-splitting wins only when one vertex carries >~1e8 paths"); ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/crossover_max_vertex_work.png", dpi=130); plt.show()
first = df[ratio < 1].raw_work_sum.mul(df[ratio < 1].work_max_share).min()
RES["crossover"] = dict(n_graphs_hubsplit_faster_than_edgejoin=int((ratio < 1).sum()), smallest_busiest_vertex_work_where_hypercube_faster=float(first),
                        largest_busiest_vertex_work_where_edgejoin_faster=float(maxw[ratio >= 1].max()))
print(RES["crossover"])
""")

md("""
## 3. Classifier — which class is fastest?
Models: **majority** (always the most common training class), **rule** (fixed prior: graphs with work hubs → hubsplit, else native; not tuned),
**logistic regression**, a **depth-2 decision tree** and a **one-threshold stump on the work of the busiest vertex** (readable, suit very few hub-splitting examples), **random forest**, **extra-trees**. Three held-out schemes:

* **LOGO** – leave one *graph* out (train on all others).
* **LOSG** – leave one *structural group* out (a whole generator family, or all hub-sweep graphs with the same number of hubs, is unseen) — the harshest test.
* **SEED202** – train on everything except the hub-sweep graphs generated with seed 202, test on those (new random graphs, same recipes).
""")
code("""
def make_models(feature_cols=None):
    d = {
        "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
        "tree_depth2": lambda: DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=SEED),
        "random_forest": lambda: RandomForestClassifier(n_estimators=NTREES, min_samples_leaf=1, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        "extra_trees": lambda: ExtraTreesClassifier(n_estimators=NTREES, min_samples_leaf=1, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
    }
    if feature_cols is not None and "log_max_work" in feature_cols:
        j = list(feature_cols).index("log_max_work")   # a one-threshold rule learned on ONE input: work of the busiest vertex
        d["stump_max_work"] = lambda: make_pipeline(FunctionTransformer(lambda A, j=j: A[:, [j]]), DecisionTreeClassifier(max_depth=1, random_state=SEED))
    return d

def splits(scheme):
    n = len(df); idx = np.arange(n)
    if scheme == "LOGO":
        for i in idx: yield np.delete(idx, i), np.array([i])
    elif scheme == "LOSG":
        for g in df.group.unique():
            te = idx[(df.group == g).values]; yield np.setdiff1d(idx, te), te
    elif scheme == "SEED202":
        te = idx[((df.family == "hub_sweep") & (df.seed == 202)).values]
        if len(te): yield np.setdiff1d(idx, te), te

def rule_pred(d):  # fixed design prior
    return np.where(d.raw_work_sum * d.work_in_hubs_share > 0, np.where(d.work_in_hubs_share > 0, "hubsplit", "native"), "native")

X = df[FEATS].values
def eval_classifiers(feature_cols, tag):
    out = {}
    Xf = df[feature_cols].values
    for scheme in ["LOGO", "LOSG", "SEED202"]:
        sp = list(splits(scheme))
        if not sp: continue
        preds = {m: np.empty(len(df), dtype=object) for m in ["majority", "rule", *make_models(feature_cols)]}
        tested = np.zeros(len(df), bool)
        for tr, te in sp:
            vals, counts = np.unique(y_class[tr], return_counts=True)
            preds["majority"][te] = vals[counts.argmax()]
            preds["rule"][te] = rule_pred(df.iloc[te])
            for name, mk in make_models(feature_cols).items():
                if len(vals) == 1: preds[name][te] = vals[0]; continue
                mdl = mk().fit(Xf[tr], y_class[tr]); preds[name][te] = mdl.predict(Xf[te])
            tested[te] = True
        res = {}
        for m, p in preds.items():
            yt, yp = y_class[tested], p[tested]
            res[m] = dict(accuracy=accuracy_score(yt, yp), balanced_accuracy=balanced_accuracy_score(yt, yp),
                          macro_f1=f1_score(yt, yp, average="macro", labels=CLASSES),
                          acc_clear_only=accuracy_score(yt[df.label_clear.values[tested]], yp[df.label_clear.values[tested]]) if df.label_clear.values[tested].any() else float("nan"))
        out[scheme] = dict(n_test=int(tested.sum()), scores=res, preds={m: p for m, p in preds.items()}, tested=tested)
    return out

clf12 = eval_classifiers(FEATS, "12 features")
rows = []
for scheme, o in clf12.items():
    for m, s in o["scores"].items():
        rows.append(dict(scheme=scheme, n_test=o["n_test"], model=m, **{k: round(v, 3) for k, v in s.items()}))
clf_table = pd.DataFrame(rows)
display(clf_table.pivot(index="model", columns="scheme", values="accuracy").loc[["majority", "rule", "logreg", "tree_depth2", "stump_max_work", "random_forest", "extra_trees"]])
RES["classifier"] = {sch: {"n_test": o["n_test"], "scores": o["scores"]} for sch, o in clf12.items()}
""")
md("""
### 3b. How the classifiers compare (all three held-out schemes)
Bars: accuracy, balanced accuracy (average recall per class, so a model that only says "edge-join" scores low) and macro-F1. The dashed line is the majority-class accuracy of that scheme: a model whose accuracy bar does not pass it adds nothing over the default.
""")
code("""
MODELS = ["majority", "rule", "logreg", "tree_depth2", "stump_max_work", "random_forest", "extra_trees"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, sch in zip(axes, ["LOGO", "LOSG", "SEED202"]):
    sc = clf12[sch]["scores"]; xs = np.arange(len(MODELS)); w = 0.27
    for j, (met, lab, col) in enumerate([("accuracy", "accuracy", "#4C78A8"), ("balanced_accuracy", "balanced accuracy", "#F58518"), ("macro_f1", "macro-F1", "#72B7B2")]):
        ax.bar(xs + (j - 1) * w, [sc[m][met] for m in MODELS], w, label=lab, color=col)
    ax.axhline(sc["majority"]["accuracy"], ls="--", c="k", lw=1, label="majority accuracy")
    ax.set_xticks(xs); ax.set_xticklabels(MODELS, rotation=40, ha="right"); ax.set_ylim(0, 1.15)
    ax.set_title(f"{sch}  (n = {clf12[sch]['n_test']} held-out graphs)"); ax.grid(axis="y", alpha=.3)
axes[0].set_ylabel("score"); axes[2].legend(fontsize=7, loc="upper right", ncol=2)
plt.suptitle("Classifier comparison on held-out graphs", y=1.02); plt.tight_layout(); plt.savefig(f"{OUT}/classifier_comparison.png", dpi=130, bbox_inches="tight"); plt.show()
""")
code("""
# Per-class recall (leave-one-graph-out): which classes can each model actually find?
rec = pd.DataFrame({m: {c: float(np.mean(clf12["LOGO"]["preds"][m][y_class == c] == c)) for c in CLASSES} for m in MODELS}).T
fig, ax = plt.subplots(figsize=(5.8, 4.2))
im = ax.imshow(rec.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([f"{c}\\n(n={int((y_class == c).sum())})" for c in CLASSES]); ax.set_yticks(range(len(MODELS))); ax.set_yticklabels(MODELS)
for i in range(len(MODELS)):
    for j in range(3): ax.text(j, i, f"{rec.values[i, j]:.2f}", ha="center", va="center", color="black" if rec.values[i, j] < .6 else "white")
ax.set_title("Recall per true class (LOGO)"); plt.colorbar(im, ax=ax, fraction=.046); plt.tight_layout()
plt.savefig(f"{OUT}/classifier_recall_heatmap.png", dpi=130); plt.show(); RES["recall_logo"] = rec.round(3).to_dict(orient="index")
""")

code("""
# Ablation: do the two derived inputs (log_hub_work, log_max_work) help?  (11 base features vs 13)
clf11 = eval_classifiers(BASE, "11 features")
ab = pd.DataFrame({sch: {"11 base features (RF)": clf11[sch]["scores"]["random_forest"]["accuracy"], "13 features (RF)": clf12[sch]["scores"]["random_forest"]["accuracy"],
                         "majority": clf12[sch]["scores"]["majority"]["accuracy"]} for sch in clf12}).round(3)
display(ab); RES["ablation_rf_accuracy"] = ab.to_dict()
""")
md("""
### Where the choice actually matters: the skewed regime
Most graphs have no dominant hub, so "edge-join" is trivially right there. The informative test is the subset where one vertex holds most of the work
(`work_max_share >= 0.9`). Below, accuracy of each held-out (LOGO) prediction on that subset only, with the majority baseline *of that subset*.
""")
code("""
mask = (df.work_max_share >= 0.9).values
print(f"skewed-regime graphs: {mask.sum()}  class counts: {pd.Series(y_class[mask]).value_counts().to_dict()}")
sub = {}
for m_, p_ in clf12["LOGO"]["preds"].items():
    sub[m_] = dict(accuracy=float(accuracy_score(y_class[mask], p_[mask])), macro_f1=float(f1_score(y_class[mask], p_[mask], average="macro", labels=CLASSES)))
sub_tab = pd.DataFrame(sub).T.round(3); display(sub_tab)
RES["skewed_regime"] = dict(n=int(mask.sum()), class_counts=pd.Series(y_class[mask]).value_counts().to_dict(), classifier_logo=sub_tab.to_dict(orient="index"))
""")

code("""
# confusion matrix of the best classifier by LOGO accuracy (non-majority) + feature importance
best_name = max(["logreg", "tree_depth2", "stump_max_work", "random_forest", "extra_trees"], key=lambda m: clf12["LOGO"]["scores"][m]["accuracy"])
p = clf12["LOGO"]["preds"][best_name]
cm = confusion_matrix(y_class, p, labels=CLASSES)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
im = ax[0].imshow(cm, cmap="Blues"); ax[0].set_xticks(range(3)); ax[0].set_xticklabels(CLASSES); ax[0].set_yticks(range(3)); ax[0].set_yticklabels(CLASSES)
for i in range(3):
    for j in range(3): ax[0].text(j, i, cm[i, j], ha="center", va="center", color="black")
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true"); ax[0].set_title(f"Confusion matrix ({best_name}, leave-one-graph-out)")
rf = RandomForestClassifier(n_estimators=NTREES, class_weight="balanced_subsample", random_state=SEED, n_jobs=1).fit(X, y_class)
imp = pd.Series(rf.feature_importances_, index=FEATS).sort_values()
imp.plot.barh(ax=ax[1], color="#4C78A8"); ax[1].set_title("Random-forest feature importance (fit on all graphs)")
plt.tight_layout(); plt.savefig(f"{OUT}/classifier_confusion_importance.png", dpi=130); plt.show()
RES["best_classifier_logo"] = best_name; RES["confusion_logo"] = cm.tolist(); RES["feature_importance"] = imp.round(4).to_dict()
print("per-class recall (LOGO,", best_name, "):", {c: round(cm[i, i] / max(cm[i].sum(), 1), 3) for i, c in enumerate(CLASSES)})
""")

md("""
## 4. Runtime regressor
One multi-output random forest predicts `log(median total_s)` for all three strategies at once (this is the *Bao/Lero-style* "predict cost, then choose" idea).
Held out by LOGO and LOSG.
""")
code("""
def fit_reg(Xtr, Ytr):
    return RandomForestRegressor(n_estimators=NTREES, min_samples_leaf=1, max_features=1.0, random_state=SEED, n_jobs=1).fit(Xtr, Ytr)

def cv_reg(scheme):
    pred = np.full_like(LOGT, np.nan)
    for tr, te in splits(scheme):
        pred[te] = fit_reg(X[tr], LOGT[tr]).predict(X[te])
    return pred

PRED = {sch: cv_reg(sch) for sch in ["LOGO", "LOSG", "SEED202"] if len(list(splits(sch)))}
reg_rows = {}
for sch, P in PRED.items():
    m = ~np.isnan(P[:, 0])
    rank_rho = np.nanmean([spearmanr(P[i], LOGT[i])[0] for i in np.where(m)[0]])
    top2 = np.mean([np.argmin(LOGT[i]) in np.argsort(P[i])[:2] for i in np.where(m)[0]])
    reg_rows[sch] = dict(n=int(m.sum()), r2_log=r2_score(LOGT[m].ravel(), P[m].ravel()),
                         median_abs_pct_err=float(np.median(np.abs(np.exp(P[m]) / T[m] - 1)) * 100),
                         per_graph_spearman=float(rank_rho), top1=float(np.mean(P[m].argmin(1) == LOGT[m].argmin(1))), top2=float(top2))
reg_tab = pd.DataFrame(reg_rows).T.round(3); display(reg_tab); RES["regressor"] = reg_tab.to_dict(orient="index")
P = PRED["LOGO"]
fig, ax = plt.subplots(figsize=(4.8, 4.4))
for k, s in enumerate(STRATS): ax.scatter(np.exp(LOGT[:, k]), np.exp(P[:, k]), s=14, alpha=.7, label=s, color=colors[s])
lim = [T.min() * .8, T.max() * 1.2]; ax.plot(lim, lim, "k--", lw=.8); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("measured median total_s"); ax.set_ylabel("predicted (held-out graph)"); ax.set_title("Regressor: predicted vs measured (LOGO)"); ax.legend(fontsize=6)
plt.tight_layout(); plt.savefig(f"{OUT}/regressor_pred_vs_actual.png", dpi=130); plt.show()
""")

md("""
## 5. Selector
For a new graph: predict the runtime of each strategy and take the argmin. A **native-fallback** variant keeps native unless the predicted gain over native exceeds a margin
(here 5 %, fixed in advance), which is the "do no harm" default asked for in the plan. The selector's class is then compared with the true best class.
""")
code("""
FALLBACK = 0.05
def choose(Pm, fallback=None):
    ch = Pm.argmin(1)
    if fallback is not None:
        nat = STRATS.index(NATIVE)
        gain = 1 - np.exp(Pm.min(1) - Pm[:, nat])
        ch = np.where(gain > fallback, ch, nat)
    return ch

def policy_table(scheme):
    P = PRED[scheme]; m = ~np.isnan(P[:, 0]); ix = np.where(m)[0]
    t_best = T[ix].min(1); pol = {}
    for k, s in enumerate(STRATS): pol[f"always {s}"] = np.full(len(ix), k)
    rule_class = rule_pred(df.iloc[ix]); pol["rule (hubs->unified, else native)"] = np.array([STRATS.index("hypercube_unified__aqe_off") if c == "hubsplit" else STRATS.index(NATIVE) for c in rule_class])
    pol["ML selector"] = choose(P[ix]); pol["ML selector + native fallback"] = choose(P[ix], FALLBACK)
    pol["oracle"] = T[ix].argmin(1)
    out = []
    for name, ch in pol.items():
        t = T[ix, ch]; nat = T[ix, STRATS.index(NATIVE)]
        cls_ok = np.mean([CLASS_OF[STRATS[c]] == y_class[i] for c, i in zip(ch, ix)])
        out.append(dict(policy=name, class_accuracy=cls_ok, strategy_accuracy=float(np.mean(ch == T[ix].argmin(1))),
                        mean_regret_pct=float(np.mean(t / t_best - 1) * 100), median_regret_pct=float(np.median(t / t_best - 1) * 100),
                        worst_regret_pct=float(np.max(t / t_best - 1) * 100), geomean_speedup_vs_native=float(np.exp(np.mean(np.log(nat / t)))),
                        total_time_s=float(t.sum()), picks_slower_than_native_pct=float(np.mean(t > nat * 1.0) * 100)))
    return pd.DataFrame(out), ix, pol

POL = {}
for sch in PRED:
    tab, ix, pol = policy_table(sch); POL[sch] = (tab, ix, pol)
    display(Markdown(f"**{sch}** — {len(ix)} held-out graphs; majority-class baseline for class accuracy = {max(np.mean(y_class[ix] == c) for c in CLASSES):.3f}"))
    display(tab.round(3))
_ch = choose(PRED["LOGO"])[mask]
RES["skewed_regime"]["selector_class_accuracy"] = float(np.mean([CLASS_OF[STRATS[c]] == t for c, t in zip(_ch, y_class[mask])]))
RES["skewed_regime"]["selector_regret_pct"] = float(np.mean(T[mask][np.arange(mask.sum()), _ch] / T[mask].min(1) - 1) * 100)
RES["skewed_regime"]["always_edgejoin_regret_pct"] = float(np.mean(T[mask][:, 0] / T[mask].min(1) - 1) * 100)
print("skewed regime: selector class accuracy", round(RES["skewed_regime"]["selector_class_accuracy"], 3), "| mean regret selector", round(RES["skewed_regime"]["selector_regret_pct"], 1), "% vs always-edgejoin", round(RES["skewed_regime"]["always_edgejoin_regret_pct"], 1), "%")
RES["policies"] = {sch: POL[sch][0].round(4).to_dict(orient="records") for sch in POL}
RES["majority_class_accuracy_in_scheme"] = {sch: float(max(np.mean(y_class[POL[sch][1]] == c) for c in CLASSES)) for sch in POL}
""")

code("""
# Regret per graph (LOGO) and total-time bars
tab, ix, pol = POL["LOGO"]
show = ["always native__aqe_on", "always edgejoin__aqe_off", "rule (hubs->unified, else native)", "ML selector", "ML selector + native fallback"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw={"width_ratios": [3, 1.3]})
order = np.argsort(df.iloc[ix].work_in_hubs_share.values)
w = 0.16
for j, name in enumerate(show):
    r = (T[ix, pol[name]] / T[ix].min(1) - 1) * 100
    ax[0].bar(np.arange(len(ix)) + j * w, r[order], w, label=name)
ax[0].set_xlabel("held-out graphs, sorted by hub share of work"); ax[0].set_ylabel("regret vs best (%)"); ax[0].set_title("Regret per graph (leave-one-graph-out)"); ax[0].legend(fontsize=6); ax[0].set_xticks([])
t = tab.set_index("policy").loc[show + ["oracle"], "total_time_s"]
t.plot.barh(ax=ax[1], color="#4C78A8"); ax[1].set_xlabel("sum of runtimes (s)"); ax[1].set_title("Total time over all graphs")
plt.tight_layout(); plt.savefig(f"{OUT}/regret_and_total_time.png", dpi=130); plt.show()
""")

md("""
## 6. Overheads and sensitivity
*Inference time* is measured here. *Feature-extraction time* was not measured as a separate step; the Spark degree/work profiling that the hub strategies perform
(`profiling_s`) is the closest measured proxy, so it is used below as a pessimistic per-graph cost for obtaining the features.
""")
code("""
mdl = fit_reg(X, LOGT); t0 = time.perf_counter()
for _ in range(50): mdl.predict(X[:1])
inf_ms = (time.perf_counter() - t0) / 50 * 1000
prof = np.median([np.median(df[f"median_profiling_s__hypercube_unified__aqe_off"].dropna()), 0])
prof = float(np.median(df["median_profiling_s__hypercube_unified__aqe_off"].dropna()))
tab, ix, pol = POL["LOGO"]; t = T[ix, pol["ML selector + native fallback"]]
nat = T[ix, STRATS.index(NATIVE)]; best_fixed = T[ix].sum(0).argmin()
oh = pd.Series({"inference_ms_per_graph": inf_ms, "profiling_s_proxy_for_feature_cost": prof,
                "geomean_speedup_vs_native_ML_fallback": float(np.exp(np.mean(np.log(nat / t)))),
                "geomean_speedup_vs_native_if_features_cost_proxy_added": float(np.exp(np.mean(np.log(nat / (t + prof))))),
                "best_fixed_strategy": STRATS[best_fixed]})
display(oh); RES["overhead"] = {k: (v if isinstance(v, str) else float(v)) for k, v in oh.items()}
""")

md("""
## 6b. More charts
Data shape, label noise, the full regret distribution and where the models err.
""")
code("""
# (1) Where the classes live in feature space
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
cc = {"edgejoin": "#4C78A8", "hubsplit": "#E45756", "native": "#72B7B2"}
for c in CLASSES:
    s_ = df.best_class == c
    ax[0].scatter(df.log_max_work[s_], df.work_max_share[s_], s=36, alpha=.8, label=f"{c} (n={int(s_.sum())})", color=cc[c], edgecolor="k", linewidth=.3)
ax[0].set_xlabel("log(1 + paths through the busiest vertex)"); ax[0].set_ylabel("share of all work in the busiest vertex"); ax[0].set_title("Best class in feature space"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
for c in CLASSES:
    s_ = df.best_class == c
    ax[1].scatter(df.log_work_sum[s_], df.work_in_hubs_share[s_], s=36, alpha=.8, color=cc[c], edgecolor="k", linewidth=.3)
ax[1].set_xlabel("log(1 + total two-hop paths)"); ax[1].set_ylabel("share of work in hubs"); ax[1].set_title("...and by total work vs hub share"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/feature_space.png", dpi=130); plt.show()
""")
code("""
# (2) Speed-up of edge-join and hypercube over native, by structural group
g = df.assign(ej_vs_native=T[:, 1] / T[:, 0], hc_vs_native=T[:, 1] / T[:, 2])
order = g.groupby("group").ej_vs_native.median().sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
for a, col, ttl in [(ax[0], "ej_vs_native", "edge-join"), (ax[1], "hc_vs_native", "hub-aware hypercube")]:
    a.boxplot([g[g.group == k][col].values for k in order], tick_labels=list(order), vert=True, showfliers=True)
    a.axhline(1, ls="--", c="k", lw=1); a.set_yscale("log"); a.set_title(f"native time / {ttl} time (>1 = {ttl} faster)"); a.tick_params(axis="x", rotation=40); a.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig(f"{OUT}/speedup_by_group.png", dpi=130); plt.show()
""")
code("""
# (3) Regret distribution of every policy (leave-one-graph-out): share of graphs within x % of the best choice
tab, ix, pol = POL["LOGO"]
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for name in ["always native__aqe_on", "always edgejoin__aqe_off", "always hypercube_unified__aqe_off", "rule (hubs->unified, else native)", "ML selector", "oracle"]:
    r = np.sort((T[ix, pol[name]] / T[ix].min(1) - 1) * 100)
    ax[0].step(r, np.arange(1, len(r) + 1) / len(r), where="post", label=name, lw=2 if name == "ML selector" else 1.2)
ax[0].set_xscale("symlog", linthresh=1); ax[0].set_xlabel("regret vs best strategy (%)"); ax[0].set_ylabel("share of graphs at or below"); ax[0].set_title("Cumulative regret (leave-one-graph-out)"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
sp = tab.set_index("policy").geomean_speedup_vs_native.drop("always native__aqe_on").sort_values()
sp.plot.barh(ax=ax[1], color="#4C78A8"); ax[1].axvline(1, c="k", lw=1); ax[1].set_xlabel("geometric-mean speed-up over native"); ax[1].set_title("Speed-up over native by policy (LOGO)")
plt.tight_layout(); plt.savefig(f"{OUT}/regret_cdf_speedup.png", dpi=130); plt.show()
""")
code("""
# (4) How noisy are the labels?  gap between best and runner-up vs the spread of the runs
ranked = np.sort(T, axis=1); gap = (ranked[:, 1] / ranked[:, 0] - 1) * 100
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].hist(gap, bins=np.logspace(np.log10(max(gap.min(), .1)), np.log10(gap.max()), 22), color="#4C78A8", edgecolor="w"); ax[0].set_xscale("log")
ax[0].set_xlabel("lead of the best strategy over the runner-up (%)"); ax[0].set_ylabel("graphs"); ax[0].set_title("Label margins: many winners lead by only a few %")
for flag, col, lab in [(True, "#4C78A8", "clear label (lead > both IQRs)"), (False, "#E45756", "noise-level label")]:
    s_ = df.label_clear.values == flag
    ax[1].scatter(df.label_margin_pct[s_], df.raw_work_sum[s_], s=30, color=col, alpha=.8, label=f"{lab} (n={int(s_.sum())})")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("label margin (%)"); ax[1].set_ylabel("total two-hop work"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3); ax[1].set_title("Clear vs noisy labels")
plt.tight_layout(); plt.savefig(f"{OUT}/label_noise.png", dpi=130); plt.show()
""")
code("""
# (5) Feature correlations and regressor error
fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.25, 1]})
C = df[FEATS].corr().values
im = ax[0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1); ax[0].set_xticks(range(len(FEATS))); ax[0].set_xticklabels(FEATS, rotation=75, ha="right", fontsize=7)
ax[0].set_yticks(range(len(FEATS))); ax[0].set_yticklabels(FEATS, fontsize=7); ax[0].set_title("Input feature correlations"); plt.colorbar(im, ax=ax[0], fraction=.046)
err = (PRED["LOGO"] - LOGT).ravel() * 100
ax[1].hist(np.expm1(np.clip(err / 100, -3, 3)) * 100, bins=40, color="#72B7B2", edgecolor="w"); ax[1].axvline(0, c="k", lw=1)
ax[1].set_xlabel("prediction error of runtime (%), held-out graphs"); ax[1].set_ylabel("graph x strategy pairs"); ax[1].set_title("Regressor error distribution (LOGO)")
plt.tight_layout(); plt.savefig(f"{OUT}/correlations_and_errors.png", dpi=130); plt.show()
""")

md("""
## 7. Honest summary (auto-generated from the numbers above)
""")
code("""
lg = RES["classifier"]["LOGO"]["scores"]; pt = {r["policy"]: r for r in RES["policies"]["LOGO"]}
lines = [
 f"- **Graphs:** {RES['n_graphs']} ({RES['protocols']}); class counts {RES['class_counts']}; majority-class accuracy {RES['majority_frac_global']:.1%}.",
 f"- **Classifier (LOGO accuracy):** majority {lg['majority']['accuracy']:.1%}, rule {lg['rule']['accuracy']:.1%}, depth-2 tree {lg['tree_depth2']['accuracy']:.1%}, stump {lg['stump_max_work']['accuracy']:.1%}, logreg {lg['logreg']['accuracy']:.1%}, random forest {lg['random_forest']['accuracy']:.1%}, extra-trees {lg['extra_trees']['accuracy']:.1%}.",
 f"- **Selector (LOGO):** class accuracy {pt['ML selector']['class_accuracy']:.1%}; mean regret {pt['ML selector']['mean_regret_pct']:.1f}% vs {pt['always edgejoin__aqe_off']['mean_regret_pct']:.1f}% for always-edge-join and {pt['always native__aqe_on']['mean_regret_pct']:.1f}% for always-native; oracle 0%.",
 f"- **Speed-up vs native (geomean):** ML {pt['ML selector']['geomean_speedup_vs_native']:.2f}x, ML+fallback {pt['ML selector + native fallback']['geomean_speedup_vs_native']:.2f}x, always edge-join {pt['always edgejoin__aqe_off']['geomean_speedup_vs_native']:.2f}x, oracle {pt['oracle']['geomean_speedup_vs_native']:.2f}x.",
]
display(Markdown("\\n".join(lines)))
with open(f"{OUT}/results.json", "w") as f: json.dump(RES, f, indent=1, default=float)
clf_table.to_csv(f"{OUT}/classifier_scores.csv", index=False)
pd.concat({sch: POL[sch][0] for sch in POL}).to_csv(f"{OUT}/policy_scores.csv")
pd.DataFrame({"graph_id": df.graph_id, "group": df.group, "true_class": y_class, "true_best": y_strat,
              **{f"pred_class_logo_{m}": clf12["LOGO"]["preds"][m] for m in clf12["LOGO"]["preds"]}}).to_csv(f"{OUT}/cv_predictions_logo.csv", index=False)
print("saved to", OUT)
""")

md("""
## 8. Reproducibility check
Everything above is deterministic: fixed random seeds, `n_jobs=1`, no sampling. Re-running the notebook top to bottom (300 trees, the committed `combined_table.csv`, the library versions printed at the top) must give the same numbers. The cell below compares the key results with `expected_results.json`, committed next to the notebook. To refresh the file after you deliberately change the data or code, run with `NB_UPDATE_EXPECTED=1`.
""")
code("""
def key_numbers():
    k = {}
    for sch, o in RES["classifier"].items():
        for m, sc in o["scores"].items():
            k[f"clf_acc|{sch}|{m}"] = sc["accuracy"]; k[f"clf_bal|{sch}|{m}"] = sc["balanced_accuracy"]
    for sch, rows in RES["policies"].items():
        for r in rows:
            k[f"pol_regret|{sch}|{r['policy']}"] = r["mean_regret_pct"]; k[f"pol_speedup|{sch}|{r['policy']}"] = r["geomean_speedup_vs_native"]
    for sch, r in RES["regressor"].items():
        k[f"reg_r2|{sch}"] = r["r2_log"]; k[f"reg_top2|{sch}"] = r["top2"]
    k["n_graphs"] = RES["n_graphs"]; k["majority_frac"] = RES["majority_frac_global"]
    return k
KEYS = key_numbers()
EXP = os.path.join(REPO, "notebooks", "expected_results.json")
if os.environ.get("NB_UPDATE_EXPECTED") == "1":
    json.dump({"trees": NTREES, "values": KEYS}, open(EXP, "w"), indent=1, sort_keys=True); print("expected results written:", len(KEYS), "numbers")
elif NTREES == 300 and os.path.exists(EXP):
    exp = json.load(open(EXP))["values"]
    bad = {k: (exp[k], KEYS.get(k)) for k in exp if k not in KEYS or abs(exp[k] - KEYS[k]) > 1e-6}
    print(f"REPRODUCED: all {len(exp)} key numbers match expected_results.json" if not bad else f"MISMATCH in {len(bad)} numbers, e.g. {list(bad.items())[:3]}")
    RES["reproduced"] = not bad
else:
    print("check skipped (needs NB_TREES=300 and notebooks/expected_results.json)")
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selector.ipynb")
nbf.write(nb, path)
print("wrote", path)
