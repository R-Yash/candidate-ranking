import os
from collections import defaultdict
import pickle

import networkx as nx
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from rapidfuzz import process, fuzz

from docx import Document

RAW_TOP_K = 2000
TOP_K = 500
RRF_K = 60
GRAPH_HOP2_DISCOUNT = 0.4
COOCCUR_MIN_WEIGHT = 5

client = QdrantClient(path="./embeddings")

with open("artifacts/jd_parsed.pkl", "rb") as f:
    jd_parsed: dict = pickle.load(f)

with open("artifacts/jd_embedding.pkl", "rb") as f:
    jd_vector = pickle.load(f)

_graph: nx.Graph | None = None
with open("artifacts/graph.pkl", "rb") as f:
    _graph = pickle.load(f)

def dense_retrieve(chunk_type: str, raw_top_k: int) -> list[tuple[str, float]]:
    results = client.query_points(
        collection_name="candidates",
        query=jd_vector,
        query_filter=Filter(must=[FieldCondition(key="chunk_type", match=MatchValue(value=chunk_type))]),
        limit=raw_top_k,
    ).points
    return [(point.payload["candidate_id"], point.score) for point in results]

def graph_retrieve() -> list[tuple[str, float]]:
    jd_skills = [
        s.lower()
        for s in jd_parsed.get("required_skills", []) + jd_parsed.get("preferred_skills", [])
    ]
    skill_nodes = [n for n, d in _graph.nodes(data=True) if d.get("node_type") == "skill"]
 
    seed_nodes: set[str] = set()
    for skill in jd_skills:
        match = process.extractOne(skill, skill_nodes, scorer=fuzz.token_sort_ratio, score_cutoff=85)
        if match:
            seed_nodes.add(match[0])
 
    scores: dict[str, float] = defaultdict(float)
    for seed in seed_nodes:
        for neighbor in _graph.neighbors(seed):
            ndata = _graph.nodes[neighbor]
            edge = _graph[seed][neighbor]
            if ndata.get("node_type") == "candidate":
                scores[neighbor] += edge.get("weight", 1.0)
            elif (
                ndata.get("node_type") == "skill"
                and edge.get("rel") == "CO_OCCURS"
                and edge.get("weight", 0) >= COOCCUR_MIN_WEIGHT
            ):
                for hop2 in _graph.neighbors(neighbor):
                    if _graph.nodes[hop2].get("node_type") == "candidate":
                        scores[hop2] += (
                            _graph[neighbor][hop2].get("weight", 1.0)
                            * edge["weight"]
                            * GRAPH_HOP2_DISCOUNT
                        )
 
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def retrieve(top_k: int = TOP_K, raw_top_k: int = RAW_TOP_K, weights: dict | None = None, rrf_k: int = RRF_K) -> list[dict]:
    weights = weights or {"profile_summary": 1.0, "skills": 1.0, "career_history": 1.0, "graph": 1.0}
    rrf_scores: dict[str, float] = defaultdict(float)
 
    for chunk_type in ["profile_summary", "skills", "career_history"]:
        ranked = dense_retrieve(chunk_type, raw_top_k)
        w = weights.get(chunk_type, 1.0)
        for rank, (cid, _) in enumerate(ranked, start=1):
            rrf_scores[cid] += w / (rrf_k + rank)
 
    graph_ranked = graph_retrieve()
    w = weights.get("graph", 1.0)
    for rank, (cid, _) in enumerate(graph_ranked, start=1):
        rrf_scores[cid] += w / (rrf_k + rank)
 
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"candidate_id": cid, "rrf_score": score} for cid, score in ranked]

if __name__ == "__main__":
    results = retrieve(weights={"profile_summary": 0.6, "skills": 0.65, "career_history": 0.75, "graph": 1.0})

    for r in results:
        print(f"{r['candidate_id']}\t{r['rrf_score']:.4f}")
 
    client.close()