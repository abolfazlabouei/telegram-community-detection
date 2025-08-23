import pandas as pd
import networkx as nx
from pathlib import Path

# ========= تنظیمات ورودی/خروجی =========
INPUT_PATH = "results/telegram_graph.edgelist"  # مسیر فایل edgelist: ستون‌ها: source target weight (جداکننده فضای خالی/تب)
OUT_DIR = Path("results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
TOPK = 1000
WEIGHT = "weight"

# ========= خواندن edgelist و ساخت گراف =========
# اگر فایل‌ات با space/tab جدا شده:
edges = pd.read_csv(
    INPUT_PATH, 
    sep=r"\s+", header=None, names=["source","target","weight"],
    dtype={"source":int, "target":int, "weight":float}
)

# اگر احتمال تکرار یال بین دو نود هست، وزن‌ها را جمع بزنیم (برای گرافِ بدون‌جهت):
edges['u'] = edges[['source','target']].min(axis=1)
edges['v'] = edges[['source','target']].max(axis=1)
edges = edges.groupby(['u','v'], as_index=False)['weight'].sum().rename(columns={'u':'source','v':'target'})

# گراف بدون‌جهت وزن‌دار
G = nx.Graph()
G.add_weighted_edges_from(edges[['source','target','weight']].itertuples(index=False, name=None))

# ========= 1) 1000 نود با بیشترین degree (بدون وزن) =========
deg = dict(G.degree())  # degree ساده
df_degree = (
    pd.DataFrame([{'node':n, 'degree':deg[n]} for n in G.nodes()])
    .sort_values(['degree','node'], ascending=[False, True])
    .head(TOPK)
    .reset_index(drop=True)
)
df_degree.insert(0, 'rank', range(1, len(df_degree)+1))
df_degree.to_csv(OUT_DIR / "top1000_degree_nodes.csv", index=False)

# ========= 2) 1000 یال با بیشترین weight =========
# از خود DataFrame edges استفاده می‌کنیم
df_edges_heavy = edges.sort_values('weight', ascending=False).head(TOPK).reset_index(drop=True)
df_edges_heavy.insert(0, 'rank', range(1, len(df_edges_heavy)+1))
df_edges_heavy.to_csv(OUT_DIR / "top1000_heaviest_edges.csv", index=False)

# ========= 3) 1000 نود هاب با معیار سوم: PageRank وزن‌دار =========
pr = nx.pagerank(G, weight=WEIGHT)  # power-iteration
df_pr = (
    pd.DataFrame([{'node':n, 'pagerank':pr[n]} for n in G.nodes()])
    .sort_values(['pagerank','node'], ascending=[False, True])
    .head(TOPK)
    .reset_index(drop=True)
)
df_pr.insert(0, 'rank', range(1, len(df_pr)+1))
df_pr.to_csv(OUT_DIR / "top1000_pagerank_nodes.csv", index=False)

# ========= (اختیاری) اگر strength هم خواستی =========
strength = {u: sum(d.get(WEIGHT, 1.0) for _,_,d in G.edges(u, data=True)) for u in G.nodes()}
df_strength = (
    pd.DataFrame([{'node':u, 'strength':strength[u]} for u in G.nodes()])
    .sort_values(['strength','node'], ascending=[False, True])
    .head(TOPK)
    .reset_index(drop=True)
)
df_strength.insert(0, 'rank', range(1, len(df_strength)+1))
# ذخیره فقط اگر لازم داری:
# df_strength.to_csv(OUT_DIR / "top1000_strength_nodes.csv", index=False)

print("Done:",
      (OUT_DIR / "top1000_degree_nodes.csv").as_posix(),
      (OUT_DIR / "top1000_heaviest_edges.csv").as_posix(),
      (OUT_DIR / "top1000_pagerank_nodes.csv").as_posix(),
      sep="\n")
