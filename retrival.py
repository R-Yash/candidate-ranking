from collections import defaultdict
 
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterOperator, FilterCondition
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en", 
                cache_folder="./models",
                embed_batch_size=64
            )

client = QdrantClient(path="./embeddings")
vector_store = QdrantVectorStore(client=client, collection_name='candidates')
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

def retrieve(jd_text, top_k=100, filters=None):
    retriever = index.as_retriever(similarity_top_k=2000, filters=filters)
    nodes = retriever.retrieve(jd_text)

    best_score = defaultdict(float)
    best_node = {}
    for n in nodes:
        cid = n.node.metadata["candidate_id"]
        if n.score > best_score[cid]:
            best_score[cid] = n.score
            best_node[cid] = n

    ranked = sorted(best_score.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "candidate_id": cid,
            "score": score,
            "matched_chunk_type": best_node[cid].node.metadata["chunk_type"],
            "matched_text": best_node[cid].node.text,
        }
        for cid, score in ranked
    ]

# TODO: Metadata filtering