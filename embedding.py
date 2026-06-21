import json
import faiss
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from chunking import chunk

data = json.loads(open("data/sample_candidates.json").read())
docs = chunk(data)

embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en", 
                cache_folder="./models",
                embed_batch_size=64
            ) 

faiss_index = faiss.IndexFlatIP(384)
vector_store = FaissVectorStore(faiss_index=faiss_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)

index.storage_context.persist(persist_dir="./storage")
