import os
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    StorageContext, 
    load_index_from_storage
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Set OpenAI API Key (Ensure you have already set it in environment 
variables = {}
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Replace with 
#your actual API key if needed

# 1️⃣ Load Documents from 'data/' folder
if not os.path.exists("data"):
    os.makedirs("data")
    with open("data/sample.txt", "w") as f:
        f.write("""This is a test document for Retrieval-Augmented 
Generation (RAG).""")

documents = SimpleDirectoryReader("data").load_data()

# 2️⃣ Create the Index with OpenAI Embeddings
embed_model = OpenAIEmbedding()
index = VectorStoreIndex.from_documents(documents, 
embed_model=embed_model)

# 3️⃣ Persist the Index
index.storage_context.persist("index_storage")
print("✅ Indexing completed successfully!")

# 4️⃣ Load Index from Storage
storage_context = StorageContext.from_defaults(persist_dir="index_storage")
index = load_index_from_storage(storage_context)

# 5️⃣ Create Query Engine
query_engine = index.as_query_engine()

# 6️⃣ Run a Test Query
query = "Summarize the contents of the indexed documents"
response = query_engine.query(query)

# 7️⃣ Display the Query Result
print("\n🔍 Query: ", query)
print("📄 Response: ", response)

