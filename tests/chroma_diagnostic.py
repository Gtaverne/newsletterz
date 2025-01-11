from chromadb import HttpClient
from chromadb.config import Settings
from rich.console import Console
from rich.table import Table
import json

def inspect_chroma():
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
        
        # Get collection
        collection = client.get_collection("emails")
        
        # Get collection stats
        console.print("\n[bold green]Collection Info:[/bold green]")
        console.print(f"Name: emails")
        
        # Get a sample of documents
        results = collection.get(limit=5)
        
        # Print total count
        console.print(f"\nTotal documents: {len(collection.get()['ids'])}")
        
        # Show sample metadata structure
        if results['metadatas']:
            console.print("\n[bold green]Sample Metadata Structure:[/bold green]")
            console.print(json.dumps(results['metadatas'][0], indent=2))
        
        # Create table for sample documents
        table = Table(title="Sample Documents")
        table.add_column("ID", style="cyan")
        table.add_column("Subject", style="green")
        table.add_column("From", style="blue")
        table.add_column("Preview", style="white")
        
        for i in range(len(results['ids'])):
            table.add_row(
                results['ids'][i],
                results['metadatas'][i].get('subject', 'N/A'),
                results['metadatas'][i].get('from', 'N/A'),
                results['documents'][i][:100] + "..." if results['documents'][i] else 'N/A'
            )
        
        console.print("\n[bold green]Sample Entries:[/bold green]")
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    inspect_chroma()