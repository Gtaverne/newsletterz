from chromadb import HttpClient
from chromadb.config import Settings
import httpx
from typing import List, Dict, Optional
import json

class EmailSearcher:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma = HttpClient(
            host="localhost",
            port=8183,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        self.collection = self.chroma.get_collection("emails")
        
        # Initialize Ollama client for embeddings
        self.ollama_url = "http://localhost:11434/api/embeddings"
        self.ollama_client = httpx.Client(timeout=30.0)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for search query"""
        response = self.ollama_client.post(
            self.ollama_url,
            json={"model": "mxbai-embed-large", "prompt": text}
        )
        return response.json()["embedding"]

    def search(self, 
               query: str, 
               n_results: int = 5, 
               from_domain: Optional[str] = None,
               include_relevance: bool = True) -> List[Dict]:
        """
        Search emails using vector similarity and optional domain filter
        
        Args:
            query: Search query text
            n_results: Number of results to return
            from_domain: Filter emails from specific domain (e.g., "mckinsey.com")
            include_relevance: Whether to include distance scores
            
        Returns:
            List of matching email documents with metadata
        """
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Prepare where clause for domain filtering
        where = None
        if from_domain:
            where = {"from": {"$in": [f"@{from_domain}", f"<{from_domain}>"]}}
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            result = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
            }
            if include_relevance:
                # Normalize the distance score to a 0-1 range
                # Lower distance means higher similarity
                distance = results['distances'][0][i]
                result['relevance_score'] = 1 / (1 + abs(distance))  
            formatted_results.append(result)
            
        return formatted_results

if __name__ == "__main__":
    # Example usage
    searcher = EmailSearcher()
    
    # Search for cloud computing in McKinsey emails
    print("\nSearching for cloud computing trends in Deloitte emails...")
    results = searcher.search(
        "Climate change trends from Deloitte",
        # from_domain="email.mckinsey.com",  # Updated to match the exact domain
        n_results=5
    )
    print("\nResults:")
    
    for r in results:
        print(f"\nScore: {r.get('relevance_score', 'N/A'):.3f}")
        print(f"From: {r['metadata']['from']}")
        print(f"Subject: {r['metadata']['subject']}")
        print(f"Date: {r['metadata']['date']}")
        print("Preview:", r['content'][:200], "...")