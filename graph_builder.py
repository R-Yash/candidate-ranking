import json
import math
import os
import pickle
from collections import Counter
from datetime import datetime
from itertools import combinations

import networkx as nx

PROFICIENCY_SCORE = {"beginner": 0.3, "intermediate": 0.6, "advanced": 0.9, "expert": 1.0}
COOCCUR_THRESHOLD = 10 

def build_graph(data: list[dict]) -> nx.Graph:
    G = nx.Graph()
    skill_sets: dict[str, list[str]] = {}

    for candidate in data:
        cid = candidate["candidate_id"]
        G.add_node(cid, node_type="candidate")

        candidate_skills = []
        for skill in candidate.get("skills", []):
            skill_name = skill["name"].lower().strip()
            if not G.has_node(skill_name):
                G.add_node(skill_name, node_type="skill")

            prof = PROFICIENCY_SCORE.get(skill["proficiency"], 0.5)
            weight = prof * math.log1p(skill.get("duration_months", 1))
            G.add_edge(cid, skill_name, weight=weight, rel="HAS_SKILL")
            candidate_skills.append(skill_name)

        skill_sets[cid] = candidate_skills

        for job in candidate.get("career_history", []):
            company = job["company"].lower().strip()
            industry = job.get("industry", "").lower().strip()

            if not G.has_node(company):
                G.add_node(company, node_type="company")

            if industry:
                if not G.has_node(industry):
                    G.add_node(industry, node_type="industry")
  
                if not G.has_edge(company, industry):
                    G.add_edge(company, industry, weight=1.0, rel="IN_INDUSTRY")

            if job.get("is_current") or job.get("end_date") is None:
                years_since_end = 0.0
            else:
                end = datetime.strptime(job["end_date"], "%Y-%m-%d")
                years_since_end = (datetime.now() - end).days / 365.25

            recency = 1.0 / (1.0 + years_since_end)
            edge_weight = job.get("duration_months", 1) * recency

            if G.has_edge(cid, company):
                G[cid][company]["weight"] += edge_weight
            else:
                G.add_edge(cid, company, weight=edge_weight, rel="WORKED_AT")

    cooccur: Counter = Counter()
    for skills in skill_sets.values():
        for s1, s2 in combinations(sorted(skills), 2):
            cooccur[(s1, s2)] += 1

    cooccur_added = 0
    for (s1, s2), count in cooccur.items():
        if count >= COOCCUR_THRESHOLD:
            G.add_edge(s1, s2, weight=float(count), rel="CO_OCCURS")
            cooccur_added += 1

    node_counts = Counter(d["node_type"] for _, d in G.nodes(data=True))
    print(f"Nodes — candidates: {node_counts['candidate']}, skills: {node_counts['skill']}, "
          f"companies: {node_counts['company']}, industries: {node_counts['industry']}")
    print(f"Edges — total: {G.number_of_edges()}, CO_OCCURS added: {cooccur_added}")

    return G


if __name__ == "__main__":
    with open("data/candidates.jsonl") as f:
        data = [json.loads(line) for line in f]

    G = build_graph(data)

    with open("artifacts/graph.pkl", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)