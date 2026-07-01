# Candidate Ranking

Ranks the top 100 candidates from a candidate pool against a job description, using dense retrieval (Qdrant), a skill co-occurrence knowledge graph (NetworkX), RRF fusion, and a weighted scoring function with honeypot and keyword-stuffer detection.

## Repo structure

```
rank.py                     # Produces the submission CSV
retrival.py                 # Handles dense + graph retrieval
download_embeddings.py      # Download precomputed embeddings stored in HF
precompute/
  chunking.py               # Splits a candidate into profile/skills/career_history chunks
  embedding.py              # Embeds candidate chunks into the local Qdrant store (./embeddings)
  graph_builder.py          # Builds the skill/company/industry knowledge graph
  jd_parsing.py             # Parses the job description into structured requirements

artifacts/                  # Precomputed artifacts
  jd_parsed.pkl             # Parsed JD (required/preferred skills, experience range, etc.)
  jd_embedding.pkl          # JD embedding vector
  graph.pkl                 # Knowledge graph
  skill_vocab.pkl           # List of all skills in the dataset

data/
  candidates.jsonl          # Candidate dataset to rank on
```

## Installation

1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Download precomputed embeddings from HF
```bash
python download_embeddings.py
```
## Ranking

Run the `rank.py` script to perform ranking on the candidate dataset and store results in a csv file
```bash
python rank.py submissiom.csv
```

## Architectural Decisions

### Precomputation

Since it was mentuoned in the submission spec that Job description remains static for the purposes of this challenge, It was decided to precompute and store embeddings for the JD. The `artifacts/` directory stores these embeddings and other details related to the JD. A knowledge graph built from the candidate dataset is also stored.

### Scoring

Final score is a weighted sum of RRF retrieval score (0.30), required skill coverage (0.20), preferred skill coverage (0.10), years-of-experience fit (0.10), location (0.05) and a behavioral signals. These take into account redbrob singals like recency, recruiter response rate, notice period, GitHub activity, open-to-work flag etc associated with each candidate. This score is then scaled down by a keyword-stuffer penalty when matched skills lack assessment scores or real duration behind them. Candidates that are consulting-only, have zero required-skill coverage or trip the impossibility heuristics are excluded before scoring.