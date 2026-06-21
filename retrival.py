from collections import defaultdict
 
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from docx import Document

RAW_TOP_K = 2000
TOP_K = 100

embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en", 
                cache_folder="./models",
                embed_batch_size=64
            )

client = QdrantClient(path="./embeddings")
vector_store = QdrantVectorStore(client=client, collection_name='candidates')
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

def retrieve_chunk_type(jd_text, chunk_type, raw_top_k):
    type_filter = MetadataFilters(filters=[MetadataFilter(key="chunk_type", value=chunk_type, operator=FilterOperator.EQ)])
    nodes = index.as_retriever(similarity_top_k=raw_top_k, filters=type_filter).retrieve(jd_text)
 
    best = {}
    for n in nodes:
        cid = n.node.metadata["candidate_id"]
        if cid not in best or n.score > best[cid].score:
            best[cid] = n
 
    return sorted(best.values(), key=lambda n: n.score, reverse=True)

def retrieve(jd_text, filters=None, top_k=TOP_K, raw_top_k=RAW_TOP_K, weights=None, rrf_k=60):
    weights = weights or {ct: 1.0 for ct in ["profile_summary", "skills", "career_history"]}
 
    rrf_scores = defaultdict(float)
    contributions = defaultdict(dict)
    for chunk_type in ["profile_summary", "skills", "career_history"]:
        for rank, n in enumerate(retrieve_chunk_type(jd_text, chunk_type, raw_top_k), start=1):
            cid = n.node.metadata["candidate_id"]
            rrf_scores[cid] += weights[chunk_type] / (rrf_k + rank)
            contributions[cid][chunk_type] = {"score": n.score, "rank": rank, "text": n.node.text}
 
    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"candidate_id": cid, "rrf_score": score, "contributions": contributions[cid]} for cid, score in ranked]

if __name__ == "__main__":
    jd = Document('data/job_description.docx')
    jd_text = '\n'.join([paragraph.text for paragraph in jd.paragraphs])

    for r in retrieve(jd_text, weights= {"profile_summary":0.5, "skills":0.65, "career_history":0.75}):
        print(f"{r['candidate_id']}\t{r['rrf_score']:.4f}\t{list(r['contributions'])}")
 
    client.close()