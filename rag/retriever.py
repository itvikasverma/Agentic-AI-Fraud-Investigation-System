import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        try:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            # test connection
            self.client.get_collections()
        except:
            print("Falling back to local disk Qdrant for testing...")
            self.client = QdrantClient(path="./qdrant_data") # Fallback
        
        self.collection_name = 'fraud_knowledge_base'
        
    def get_all_docs_for_bm25(self):
        try:
            scroll_res = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )
            return scroll_res[0]
        except Exception:
            return []

    def retrieve(self, query: str, top_k: int = 3):
        # Semantic Search (Dense)
        query_vector = self.embedding_model.encode(query).tolist()
        try:
            dense_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k * 2
            )
        except Exception:
            dense_results = []
            
        # BM25 Search (Sparse)
        all_docs = self.get_all_docs_for_bm25()
        bm25_results = []
        if all_docs:
            tokenized_corpus = [doc.payload['content'].lower().split() for doc in all_docs]
            bm25 = BM25Okapi(tokenized_corpus)
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Get top K from BM25
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k*2]
            bm25_results = [all_docs[i] for i in top_bm25_idx if bm25_scores[i] > 0]
            
        # Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        for rank, res in enumerate(dense_results):
            if res.id not in rrf_scores:
                rrf_scores[res.id] = {"doc": res, "score": 0}
            rrf_scores[res.id]["score"] += 1.0 / (60 + rank)
            
        for rank, res in enumerate(bm25_results):
            if res.id not in rrf_scores:
                rrf_scores[res.id] = {"doc": res, "score": 0}
            rrf_scores[res.id]["score"] += 1.0 / (60 + rank)
            
        fused_docs = [item["doc"] for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)]
        
        # Cross-Encoder Reranking
        if not fused_docs:
            return []
            
        cross_inp = [[query, doc.payload['content']] for doc in fused_docs]
        cross_scores = self.reranker.predict(cross_inp)
        
        # Add rerank scores and convert to dict
        reranked = []
        for i, doc in enumerate(fused_docs):
            reranked.append({
                "id": doc.id,
                "payload": doc.payload,
                "rerank_score": float(cross_scores[i])
            })
            
        # Sort by rerank score
        reranked = sorted(reranked, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
