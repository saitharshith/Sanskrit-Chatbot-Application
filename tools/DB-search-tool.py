import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# NOTE: Update these paths to the absolute paths on your local machine
DB1_DIR = r"artifacts\chroma_vedic_db_mahabharata"
DB2_DIR = r"artifacts\chroma_vedic_db-1"
COLLECTION_NAME = "vedic_epic_collection"

try:
    # Load the embedding model into memory once
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

    # Connect to Database 1 (Gita, Ramayana, Niti Shlokas)
    vector_store_1 = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB1_DIR
    )

    # Connect to Database 2 (Mahabharata Semantic Chunks)
    vector_store_2 = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=DB2_DIR
    )
    print("✅ Ancient Scriptures Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not connect to local databases. Error: {e}")


@tool
def query_local_epic_db(query: str) -> str:
    """
    Use this tool FIRST whenever the user asks about Vedic philosophies, stories,
    Sanskrit Shlokas, the Bhagavad Gita, the Ramayana, the Mahabharata, or Niti Shlokas.
    Input: The specific concept, character, or question the user is asking about.
    """
    try:
        docs_from_db1 = vector_store_1.similarity_search(query, k=2)
        docs_from_db2 = vector_store_2.similarity_search(query, k=2)

        # 2. Merge the results
        combined_docs = docs_from_db1 + docs_from_db2

        if not combined_docs:
            return "The local scriptures do not contain specific verses regarding this query. You may need to rely on your innate knowledge or search the web."

        # 3. Format the output beautifully so the LLM knows exactly what it is reading
        formatted_results = []
        for i, doc in enumerate(combined_docs):
            # Extracting the metadata we attached during ingestion
            source = doc.metadata.get("source_text", "Unknown Epic")
            chapter = doc.metadata.get("chapter", "Unknown Chapter")

            formatted_results.append(
                f"--- Excerpt {i+1} ---\n"
                f"Source: {source} | Section: {chapter}\n"
                f"Text:\n{doc.page_content}\n"
            )

        # 4. Return the combined text to the Agent State
        final_output = "\n".join(formatted_results)
        return f"Database Retrieval Successful. Read the following excerpts to formulate your answer:\n\n{final_output}"

    except Exception as e:
        return f"My memory of the texts is currently clouded. Error accessing database: {str(e)}"