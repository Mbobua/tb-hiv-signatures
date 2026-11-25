"""Stage 6 - protein-interaction network and hub genes (Objective 2).

The top differentially expressed genes are sent to the STRING API; the returned
interactions are loaded into NetworkX, and hub genes are ranked by degree and betweenness.
If STRING is unreachable, a co-expression network (|Spearman rho| above a cut-off) is used
instead so the stage still runs, and the notebook says which one was used.
"""

from __future__ import annotations

import io
import urllib.parse
import urllib.request

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from .plots import PALETTES, dodge_labels, savefig

STRING_API = "https://string-db.org/api/tsv/network"


def string_edges(genes, species=9606, min_score=700, timeout=60) -> pd.DataFrame:
    data = urllib.parse.urlencode({
        "identifiers": "\r".join(genes),
        "species": species,
        "required_score": min_score,
        "caller_identity": "strathmore_bioinformatics_tb_hiv",
    }).encode()
    with urllib.request.urlopen(urllib.request.Request(STRING_API, data=data), timeout=timeout) as r:
        txt = r.read().decode()
    df = pd.read_csv(io.StringIO(txt), sep="\t")
    df = df.rename(columns={"preferredName_A": "a", "preferredName_B": "b", "score": "weight"})
    edges = parse_string_edges(df)
    # STRING answers with its current HGNC names (e.g. H4C6), while the array annotation may use
    # older aliases (HIST1H4H). Map them back so nodes keep their DE results and are not duplicated.
    try:
        back = string_name_map(genes, species, timeout)
        edges[["a", "b"]] = edges[["a", "b"]].replace(back)
    except Exception:
        pass
    return edges


def string_name_map(genes, species=9606, timeout=60) -> dict:
    """STRING preferred name -> the symbol we queried (only where they differ)."""
    data = urllib.parse.urlencode({
        "identifiers": "\r".join(genes), "species": species, "limit": 1, "echo_query": 1,
        "caller_identity": "strathmore_bioinformatics_tb_hiv",
    }).encode()
    url = STRING_API.replace("/network", "/get_string_ids")
    with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=timeout) as r:
        ids = pd.read_csv(io.StringIO(r.read().decode()), sep="\t")
    ids = ids[ids["queryItem"] != ids["preferredName"]].drop_duplicates("preferredName")
    return dict(zip(ids["preferredName"], ids["queryItem"]))


def parse_string_edges(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"preferredName_A": "a", "preferredName_B": "b", "score": "weight"})
    df = df[["a", "b", "weight"]].copy()
    df["weight"] = df["weight"].astype(float)
    # STRING may return each pair twice (A-B, B-A)
    key = df.apply(lambda r: tuple(sorted((r["a"], r["b"]))), axis=1)
    return df.loc[~key.duplicated()].reset_index(drop=True)


def coexpression_edges(expr: pd.DataFrame, genes, min_abs_rho=0.7) -> pd.DataFrame:
    genes = [g for g in genes if g in expr.index]
    rho = expr.loc[genes].T.rank().corr().values
    iu = np.triu_indices(len(genes), k=1)
    keep = np.abs(rho[iu]) >= min_abs_rho
    return pd.DataFrame({"a": np.array(genes)[iu[0][keep]], "b": np.array(genes)[iu[1][keep]],
                         "weight": np.abs(rho[iu][keep])})


def build_network(genes, de_res: pd.DataFrame | None = None, expr=None, species=9606, min_score=700,
                  fallback_rho=0.7) -> tuple[nx.Graph, str]:
    """Returns (graph, source) where source is 'STRING' or 'co-expression'."""
    try:
        edges, source = string_edges(genes, species, min_score), "STRING"
    except Exception as e:
        if expr is None:
            raise
        print(f"STRING unavailable ({type(e).__name__}); using co-expression network instead.")
        edges, source = coexpression_edges(expr, genes, fallback_rho), "co-expression"
    G = nx.Graph()
    G.add_nodes_from(genes)
    G.add_weighted_edges_from(edges[["a", "b", "weight"]].itertuples(index=False))
    if de_res is not None:
        for g in G.nodes:
            if g in de_res.index:
                G.nodes[g]["log2FC"] = float(de_res.loc[g, "log2FC"])
    G.graph["source"] = source
    return G, source


def hub_table(G: nx.Graph, top=15) -> pd.DataFrame:
    H = G.subgraph([n for n in G.nodes if G.degree(n) > 0]).copy()
    if H.number_of_nodes() == 0:
        return pd.DataFrame(columns=["degree", "betweenness", "closeness", "community", "log2FC"])
    deg = dict(H.degree())
    btw = nx.betweenness_centrality(H, normalized=True)
    clo = nx.closeness_centrality(H)
    comms = nx.algorithms.community.greedy_modularity_communities(H)
    comm = {n: i + 1 for i, c in enumerate(sorted(comms, key=len, reverse=True)) for n in c}
    t = pd.DataFrame({"degree": deg, "betweenness": btw, "closeness": clo, "community": comm})
    t["log2FC"] = [H.nodes[n].get("log2FC", np.nan) for n in t.index]
    t = t.sort_values(["degree", "betweenness"], ascending=False)
    t.index.name = "gene"
    return t.head(top) if top else t


def network_summary(G: nx.Graph) -> dict:
    H = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
    comps = sorted(nx.connected_components(H), key=len, reverse=True) if H.number_of_nodes() else []
    return {
        "source": G.graph.get("source"),
        "genes_queried": G.number_of_nodes(),
        "genes_connected": H.number_of_nodes(),
        "edges": G.number_of_edges(),
        "largest_component": len(comps[0]) if comps else 0,
        "density": nx.density(H) if H.number_of_nodes() > 1 else 0.0,
    }


def _packed_layout(H: nx.Graph, comps, seed=7, max_width=4.0, gap=0.35) -> dict:
    """Lay out each connected component on its own and pack them in rows (largest first), so small
    components sit next to the main one instead of being flung to the edges by a global spring layout."""
    pos, x_off, y_off, row_h = {}, 0.0, 0.0, 0.0
    for c in comps:
        sub = H.subgraph(c)
        p = nx.spring_layout(sub, seed=seed, k=1.5 / np.sqrt(len(c)))
        names = list(p)
        arr = np.array([p[n] for n in names], dtype=float)
        arr -= arr.min(axis=0)
        span = arr.max() if arr.max() > 0 else 1.0
        side = 0.1 + 0.3 * np.sqrt(len(c))  # bigger components get more room
        arr = arr / span * side
        w, h = arr.max(axis=0)
        if x_off > 0 and x_off + w > max_width:
            x_off, y_off, row_h = 0.0, y_off - row_h - gap, 0.0
        for n, (x, y) in zip(names, arr):
            pos[n] = (x + x_off, y + y_off - h)
        x_off += w + gap
        row_h = max(row_h, h)
    return pos


def plot_network(G: nx.Graph, hubs: pd.DataFrame, title: str, path=None, seed=7):
    H = G.subgraph([n for n in G.nodes if G.degree(n) > 0])
    fig, ax = plt.subplots(figsize=(8, 7))
    if H.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "no interactions above threshold", ha="center")
        ax.axis("off")
        return savefig(fig, path)
    comps = sorted(nx.connected_components(H), key=len, reverse=True)[:8]
    H = H.subgraph(set().union(*comps))
    pos = _packed_layout(H, comps, seed)
    lfc = np.array([H.nodes[n].get("log2FC", np.nan) for n in H.nodes])
    cols = [PALETTES["direction"]["ns"] if np.isnan(v) else
            PALETTES["direction"]["up"] if v > 0 else PALETTES["direction"]["down"] for v in lfc]
    deg = np.array([H.degree(n) for n in H.nodes])
    nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.35, width=0.7, edge_color="#898781")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=cols, node_size=30 + 260 * deg / max(deg.max(), 1),
                           edgecolors="white", linewidths=1.0)
    # label the top hubs, but at most 3 per community so one dense clique cannot take every label
    lab, per_comm = [], {}
    for n in hubs.index:
        c = hubs.loc[n, "community"] if "community" in hubs else 0
        if n in pos and per_comm.get(c, 0) < 3:
            lab.append(n)
            per_comm[c] = per_comm.get(c, 0) + 1
        if len(lab) == 10:
            break
    if lab:
        ax.autoscale_view()
        dodge_labels(ax, [pos[n][0] for n in lab], [pos[n][1] for n in lab], lab, fontsize=8)
    ax.scatter([], [], c=PALETTES["direction"]["up"], label="higher in active TB")
    ax.scatter([], [], c=PALETTES["direction"]["down"], label="lower in active TB")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    ax.set_title(f"{title}\n(node size = degree; labelled = top hubs)")
    ax.axis("off")
    fig.tight_layout()
    return savefig(fig, path)
