"""Builds notebooks/clustering.ipynb (unsupervised view of the 62-graph feature table)."""
import nbformat as nbf
from pathlib import Path
nb=nbf.v4.new_notebook(); C=[]
def md(s): C.append(nbf.v4.new_markdown_cell(s))
def code(s): C.append(nbf.v4.new_code_cell(s))

md("""# Clustering the graph feature table (Big Data Essentials, Chapter 2)

Do the 62 benchmarked graphs form natural groups when we look **only at their pre-execution features** (no runtimes, no labels)?
We cluster the 13 graph features with Spark MLlib K-Means and Gaussian Mixture, plus Ward clustering (scikit-learn), and only afterwards compare the clusters with the graph family and with the best-strategy class as an outside check.
Everything is deterministic (fixed seeds) and writes figures and `clustering_results.json` to `datasets/derived/clustering/`.""")

code("""import os, json, warnings, itertools
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

ROOT = Path.cwd()
while not (ROOT/'datasets').exists() and ROOT != ROOT.parent: ROOT = ROOT.parent
OUT = ROOT/'datasets'/'derived'/'clustering'; OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
RES = {}
plt.rcParams.update({'figure.dpi':110,'axes.spines.top':False,'axes.spines.right':False})
print('root:', ROOT)""")

md("## 1. Data: 62 graphs, 13 pre-execution features")
code("""df = pd.read_csv(ROOT/'datasets'/'derived'/'training'/'combined_table.csv')
FEATURES = ['log_num_vertices','log_num_edges','active_fraction','in_gini','out_gini','log_in_p999','log_out_p999',
            'log_work_sum','work_max_share','log_num_work_hubs','work_in_hubs_share']
df['max_work'] = df.work_max_share*df.raw_work_sum
df['log_hub_work'] = np.log1p(df.work_in_hubs_share*df.raw_work_sum)
df['log_max_work'] = np.log1p(df.max_work)
FEATURES += ['log_hub_work','log_max_work']
X = df[FEATURES].to_numpy(float)
print(X.shape, 'missing values:', int(np.isnan(X).sum()))
df['group'] = df.family
print(df.group.value_counts().to_dict()); print(df.best_class.value_counts().to_dict())
display(df[FEATURES].describe().T[['mean','std','min','max']].round(3))
RES['n_graphs']=int(len(df)); RES['n_features']=len(FEATURES)""")

md("## 2. Standardise, then choose K (elbow and silhouette)")
code("""Xs = StandardScaler().fit_transform(X)
Ks = list(range(2,11)); inertia=[]; sil=[]
for k in Ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_))
tab = pd.DataFrame({'K':Ks,'inertia':np.round(inertia,2),'silhouette':np.round(sil,4)}); display(tab)
K_BEST = int(Ks[int(np.argmax(sil))]); print('K with highest silhouette:', K_BEST)
fig,ax = plt.subplots(1,2,figsize=(9,3.2))
ax[0].plot(Ks,inertia,'o-',color='#0072B2'); ax[0].set_title('Elbow curve (standardised features)'); ax[0].set_xlabel('Number of clusters (K)'); ax[0].set_ylabel('Inertia')
ax[1].plot(Ks,sil,'o-',color='#009E73'); ax[1].axvline(K_BEST,color='grey',ls='--',lw=0.8); ax[1].set_title('Silhouette score vs K'); ax[1].set_xlabel('Number of clusters (K)'); ax[1].set_ylabel('Silhouette')
plt.tight_layout(); plt.savefig(OUT/'fig_elbow_silhouette.png',dpi=160); plt.show()
RES['k_scan']={int(k):{'inertia':float(i),'silhouette':float(s)} for k,i,s in zip(Ks,inertia,sil)}; RES['k_best']=K_BEST""")

md("## 3. Three clustering methods on the same standardised data")
code("""from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import KMeans as SparkKMeans, GaussianMixture
spark = (SparkSession.builder.master('local[2]').appName('graph-feature-clustering')
         .config('spark.ui.enabled','false').config('spark.sql.shuffle.partitions','4').getOrCreate())
spark.sparkContext.setLogLevel('ERROR')
def to_sdf(M): return spark.createDataFrame([(Vectors.dense(r.tolist()),) for r in M], ['features'])
sdf = to_sdf(Xs).cache()
K = K_BEST
spark_km  = SparkKMeans(k=K, seed=SEED, maxIter=100).fit(sdf)
spark_gmm = GaussianMixture(k=K, seed=SEED, maxIter=100).fit(sdf)
lab_km  = np.array([r.prediction for r in spark_km.transform(sdf).select('prediction').collect()])
lab_gmm = np.array([r.prediction for r in spark_gmm.transform(sdf).select('prediction').collect()])
lab_agg = AgglomerativeClustering(n_clusters=K, linkage='ward').fit_predict(Xs)
def score(lab, M=Xs):
    return dict(sizes='/'.join(str(int(c)) for c in np.sort(np.bincount(lab))[::-1]),
                silhouette=float(silhouette_score(M,lab)), davies_bouldin=float(davies_bouldin_score(M,lab)),
                calinski_harabasz=float(calinski_harabasz_score(M,lab)),
                ari_family=float(adjusted_rand_score(df.group,lab)), ari_best_class=float(adjusted_rand_score(df.best_class,lab)))
models = {'K-Means (Spark)':lab_km,'Gaussian Mixture (Spark)':lab_gmm,'Agglomerative Ward (sklearn)':lab_agg}
cmp_df = pd.DataFrame({n:score(l) for n,l in models.items()}).T
display(cmp_df.round(4)); RES['model_comparison']={n:score(l) for n,l in models.items()}; RES['K_used']=K
RES['kmeans_sklearn_ari_family']=float(adjusted_rand_score(df.group, KMeans(n_clusters=K,n_init=20,random_state=SEED).fit_predict(Xs)))""")

md("## 4. Does scaling matter, and are the clusters stable?")
code("""# raw (not scaled) vs standardised, Spark K-Means, silhouette measured in each run's own space
raw_lab = np.array([r.prediction for r in SparkKMeans(k=K,seed=SEED,maxIter=100).fit(to_sdf(X)).transform(to_sdf(X)).select('prediction').collect()])
scale_df = pd.DataFrame({'Raw (not scaled)':{'silhouette':silhouette_score(X,raw_lab),'ari_family':adjusted_rand_score(df.group,raw_lab),'ari_best_class':adjusted_rand_score(df.best_class,raw_lab)},
                         'Standardised':{'silhouette':silhouette_score(Xs,lab_km),'ari_family':adjusted_rand_score(df.group,lab_km),'ari_best_class':adjusted_rand_score(df.best_class,lab_km)}}).T
display(scale_df.round(4)); RES['scaling']=scale_df.to_dict('index')
var_share = X.var(axis=0)/X.var(axis=0).sum(); top = int(np.argmax(var_share))
print('largest raw variance share:', FEATURES[top], round(float(var_share[top])*100,2), '%'); RES['raw_top_variance_feature']=FEATURES[top]; RES['raw_top_variance_share_pct']=float(var_share[top]*100)
# stability over 20 seeds
runs=[]; sils=[]
for s in range(20):
    m = SparkKMeans(k=K, seed=s, maxIter=100).fit(sdf)
    l = np.array([r.prediction for r in m.transform(sdf).select('prediction').collect()]); runs.append(l); sils.append(silhouette_score(Xs,l))
pair = [adjusted_rand_score(a,b) for a,b in itertools.combinations(runs,2)]
print('silhouette over 20 seeds: mean %.4f sd %.4f'%(np.mean(sils),np.std(sils))); print('pairwise ARI: mean %.3f min %.3f'%(np.mean(pair),np.min(pair)))
RES['stability']={'silhouette_mean':float(np.mean(sils)),'silhouette_sd':float(np.std(sils)),'ari_mean':float(np.mean(pair)),'ari_min':float(np.min(pair))}""")

md("## 5. What do the clusters mean? (profile and outside check)")
code("""prof = pd.DataFrame(Xs, columns=FEATURES).groupby(lab_km).mean()
fig,ax = plt.subplots(figsize=(8.5,3.4)); im = ax.imshow(prof.T.values, cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
ax.set_xticks(range(K)); ax.set_xticklabels([f'cluster {i}\\n(n={int((lab_km==i).sum())})' for i in range(K)]); ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
for i in range(K):
    for j in range(len(FEATURES)): ax.text(i,j,f'{prof.values[i,j]:.1f}',ha='center',va='center',fontsize=7)
plt.colorbar(im,label='standardised mean'); plt.title('Cluster profile (Spark K-Means)'); plt.tight_layout(); plt.savefig(OUT/'fig_cluster_profile_heatmap.png',dpi=160); plt.show()
ct_family = pd.crosstab(lab_km, df.group); ct_class = pd.crosstab(lab_km, df.best_class)
print('clusters vs graph family'); display(ct_family); print('clusters vs best strategy class'); display(ct_class)
RES['crosstab_family']={int(k):{c:int(v) for c,v in r.items()} for k,r in ct_family.iterrows()}
RES['crosstab_best_class']={int(k):{c:int(v) for c,v in r.items()} for k,r in ct_class.iterrows()}
RES['profile']={int(i):{f:float(prof.values[i,j]) for j,f in enumerate(FEATURES)} for i in range(K)}
# hub-splitting graphs: where do they land?
RES['hubsplit_cluster_counts']={int(k):int(v) for k,v in pd.Series(lab_km[df.best_class.values=='hubsplit']).value_counts().items()}
print('cluster of the 6 hub-splitting graphs:', RES['hubsplit_cluster_counts'])""")

code("""pca = PCA(n_components=2, random_state=SEED); Z = pca.fit_transform(Xs); RES['pca_explained_variance']=[float(v) for v in pca.explained_variance_ratio_]
fig,axs = plt.subplots(2,2,figsize=(8,6.2))
panels=[('K-Means (Spark)',lab_km),('Gaussian Mixture (Spark)',lab_gmm),('Agglomerative Ward (sklearn)',lab_agg),('Graph family (outside check)',pd.factorize(df.group)[0])]
for a,(t,l) in zip(axs.ravel(),panels):
    a.scatter(Z[:,0],Z[:,1],c=l,cmap='viridis',s=22,edgecolor='white',lw=0.3); a.set_title(t,fontsize=9); a.set_xlabel('PC1'); a.set_ylabel('PC2')
hs = df.best_class.values=='hubsplit'
axs[0,0].scatter(Z[hs,0],Z[hs,1],facecolors='none',edgecolors='red',s=70,lw=1.1,label='hub-splitting wins'); axs[0,0].legend(fontsize=7,loc='best')
plt.tight_layout(); plt.savefig(OUT/'fig_pca_model_comparison.png',dpi=160); plt.show()
print('PCA variance explained:', np.round(pca.explained_variance_ratio_,3))""")

code("""pd.DataFrame({'graph_id':df.graph_id,'family':df.group,'best_class':df.best_class,'kmeans_spark':lab_km,'gmm_spark':lab_gmm,'ward':lab_agg}).to_csv(OUT/'cluster_assignments.csv',index=False)
json.dump(RES, open(OUT/'clustering_results.json','w'), indent=2)
spark.stop(); print('saved to', OUT)""")
nb['cells']=C
nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
nbf.write(nb, Path(__file__).with_name('clustering.ipynb')); print('ok')
