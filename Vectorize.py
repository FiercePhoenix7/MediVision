import chromadb
from chromadb.utils import embedding_functions

chunks = []
with open("Medicine_Info.md", 'r', encoding='utf-8') as f:
    lines = f.readlines()
    chunk = ''
    for line in lines:
        if line.startswith("# "):
            if chunk != '':
                chunks.append(chunk)
            chunk = line
        else:
            chunk = chunk + line

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./Vectorized_Medicine_Info")
collection = client.get_or_create_collection(
            name="Medicine_Details",
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"}  
)

count = 1
ids = []
for chunk in chunks:
    ids.append(f"Medicine {count}")
    count += 1

collection.add(
    ids=ids,
    documents=chunks
)
