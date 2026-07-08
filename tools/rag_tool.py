"""
RAG Tool — Semantic search over pollution reports using Chroma + Sentence-Transformers.
"""

import os
import sqlite3
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

EMBEDDINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "embeddings")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pollution_db.sqlite")

# Global instances
_model = None
_collection = None


def get_embedding_model():
    """Load sentence transformer model (cached)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection():
    """Get or create Chroma collection."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=EMBEDDINGS_DIR)
        _collection = client.get_or_create_collection(
            name="pollution_reports",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def index_reports():
    """Index all pollution reports into Chroma vector store."""
    collection = get_collection()
    model = get_embedding_model()
    
    # Check if already indexed
    if collection.count() > 0:
        return collection.count()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, location_name, city, pollution_type, severity, 
               description, aqi_reading, reported_at, recommended_action
        FROM pollution_reports
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return 0
    
    documents = []
    metadatas = []
    ids = []
    
    for row in rows:
        doc = (
            f"Location: {row[1]}, City: {row[2]}. "
            f"Pollution Type: {row[3]}. Severity: {row[4]}. "
            f"Description: {row[5]}. AQI: {row[6]}. "
            f"Reported: {row[7]}. Action: {row[8]}"
        )
        documents.append(doc)
        metadatas.append({
            "location": row[1],
            "city": row[2],
            "pollution_type": row[3],
            "severity": row[4],
            "aqi": row[6] or 0,
            "reported_at": row[7] or ""
        })
        ids.append(str(row[0]))
    
    # Batch insert (Chroma has batch limits)
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        embeddings = model.encode(batch_docs).tolist()
        
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
    
    return len(documents)


def semantic_search(query: str, n_results: int = 10, city_filter: str = None) -> list:
    """
    Search pollution reports semantically.
    
    Args:
        query: Natural language search query
        n_results: Number of results to return
        city_filter: Optional city filter
    
    Returns:
        List of dicts with matched documents and metadata
    """
    collection = get_collection()
    model = get_embedding_model()
    
    # Ensure indexed
    if collection.count() == 0:
        index_reports()
    
    query_embedding = model.encode([query]).tolist()
    
    where_filter = None
    if city_filter and city_filter != "All":
        where_filter = {"city": city_filter}
    
    try:
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(n_results, collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        output = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                output.append({
                    "document": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "similarity": 1 - results["distances"][0][i] if results["distances"] else 0
                })
        
        return output
        
    except Exception as e:
        return [{"document": f"Search error: {str(e)}", "metadata": {}, "similarity": 0}]


def add_report_to_index(report_id: int, document: str, metadata: dict):
    """Add a new report to the vector index."""
    collection = get_collection()
    model = get_embedding_model()
    
    embedding = model.encode([document]).tolist()
    
    collection.add(
        documents=[document],
        embeddings=embedding,
        metadatas=[metadata],
        ids=[str(report_id)]
    )


def get_similar_reports(location: str, pollution_type: str, n_results: int = 5) -> list:
    """Find similar past reports for a given location and pollution type."""
    query = f"Pollution at {location}, type: {pollution_type}"
    return semantic_search(query, n_results=n_results)
