import os
import json

from google.genai import types as genai_types
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

from precompute.chunking import chunk
from dotenv import load_dotenv
load_dotenv()

with open("data/candidates.jsonl") as f:
    data = [json.loads(line) for line in f]

docs = chunk(data)

embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-2",
    embedding_config=genai_types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT",output_dimensionality=768),
    embed_batch_size=100
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