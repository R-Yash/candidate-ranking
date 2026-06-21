from collections import defaultdict

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

vector_store = FaissVectorStore.from_persist_dir("./storage")
storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir="./storage")
index = load_index_from_storage(storage_context, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=2000)

def retrieve(jd_text, top_k=100):
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
