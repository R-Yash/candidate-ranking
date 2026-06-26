import json
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
 
from chunking import chunk

# data = json.loads(open("data/sample_candidates.json").read())
with open('data/candidates.jsonl') as f:
    data = [json.loads(line) for line in f]

docs = chunk(data)

embed_model = HuggingFaceEmbedding(
                model_name="BAAI/bge-small-en", 
                cache_folder="./models",
                embed_batch_size=128
            ) 

client = QdrantClient(path="./embeddings")
vector_store = QdrantVectorStore(client=client, collection_name="candidates")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
    embed_model=embed_model,
    show_progress=True,
)

client.close() 