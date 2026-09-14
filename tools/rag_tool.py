from langchain_core.tools import tool
from rag.retriever import HybridRetriever
import json

retriever = HybridRetriever()

@tool
def search_fraud_knowledge_base(query: str) -> str:
    """
    Searches the fraud knowledge base for policies, guidelines, and historical cases.
    Input should be a search query describing the suspicious pattern.
    Returns the most relevant policy documents and historical cases.
    """
    results = retriever.retrieve(query, top_k=3)
    if not results:
        return "No relevant documents found."
        
    formatted_results = []
    for doc in results:
        formatted_results.append({
            "source": doc["payload"].get("source", "Unknown"),
            "content": doc["payload"].get("content", ""),
            "relevance_score": doc.get('rerank_score', 0)
        })
        
    return json.dumps(formatted_results, indent=2)
