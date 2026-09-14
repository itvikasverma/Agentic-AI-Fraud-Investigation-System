import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import glob
import uuid

def ingest_docs():
    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Qdrant client
    # In local development we can just use memory or a local file if Qdrant isn't running yet.
    # For this project, we'll connect to the Qdrant instance.
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
    
    print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}...")
    try:
        client = QdrantClient(host=qdrant_host, port=qdrant_port)
        # Check if collection exists
        collections = client.get_collections().collections
        if not any(c.name == 'fraud_knowledge_base' for c in collections):
            client.create_collection(
                collection_name='fraud_knowledge_base',
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            print("Created collection 'fraud_knowledge_base'")
    except Exception as e:
        print(f"Failed to connect to Qdrant (is it running?): {e}")
        print("Falling back to local disk Qdrant for testing...")
        client = QdrantClient(path="./qdrant_data")
        
        # In local disk mode, we only need to create if it doesn't exist
        try:
            client.get_collection('fraud_knowledge_base')
        except:
            client.create_collection(
                collection_name='fraud_knowledge_base',
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    # Read documents
    doc_paths = glob.glob('docs/*.md')
    points = []
    
    for path in doc_paths:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Simple chunking by sections
            chunks = content.split('\n## ')
            for i, chunk in enumerate(chunks):
                if i > 0:
                    chunk = '## ' + chunk
                
                if not chunk.strip():
                    continue
                    
                vector = model.encode(chunk).tolist()
                
                metadata = {
                    "source": path,
                    "content": chunk,
                    "type": "case" if "case_" in path else "policy"
                }
                
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload=metadata
                    )
                )
    
    if points:
        client.upsert(
            collection_name='fraud_knowledge_base',
            points=points
        )
        print(f"Ingested {len(points)} document chunks into Qdrant.")
    else:
        print("No documents found to ingest.")

if __name__ == "__main__":
    ingest_docs()
