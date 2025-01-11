from chromadb import HttpClient
from chromadb.config import Settings
import httpx
import numpy as np
from rich.console import Console
from rich.table import Table

def analyze_distances():
    console = Console()
    
    try:
        # Initialize ChromaDB client
        client = HttpClient(
            host="localhost",
            port=8183,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        
        collection = client.get_collection("emails")
        
        # Get embedding for a test query
        response = httpx.post(
            "http://localhost:11434/api/embeddings",
            json={"model": "mxbai-embed-large", "prompt": "AI artificial intelligence machine learning"}
        )
        query_embedding = response.json()["embedding"]
        
        # Get results with distances
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=50,  # Get a good sample
            include=['metadatas', 'distances']
        )
        
        distances = results['distances'][0]
        
        # Calculate statistics
        stats = {
            'min': np.min(distances),
            'max': np.max(distances),
            'mean': np.mean(distances),
            'median': np.median(distances),
            'std': np.std(distances)
        }
        
        console.print("\n[bold green]Distance Statistics:[/bold green]")
        for key, value in stats.items():
            console.print(f"{key}: {value:.4f}")
        
        # Show sample results with distances
        table = Table(title="\nSample Results with Distances")
        table.add_column("Distance", style="cyan")
        table.add_column("Simple Score", style="green")
        table.add_column("Normalized Score", style="blue")
        table.add_column("Subject", style="white")
        
        # Calculate normalized scores
        min_dist = min(distances)
        max_dist = max(distances)
        dist_range = max_dist - min_dist
        
        for i in range(min(10, len(distances))):
            distance = distances[i]
            simple_score = 1 / (1 + abs(distance))
            
            # Min-max normalization
            normalized_score = 1 - ((distance - min_dist) / dist_range if dist_range > 0 else 0)
            
            table.add_row(
                f"{distance:.4f}",
                f"{simple_score:.4f}",
                f"{normalized_score:.4f}",
                results['metadatas'][0][i]['subject']
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    analyze_distances()