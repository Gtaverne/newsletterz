from chromadb import HttpClient
from chromadb.config import Settings

def test_chroma_connection():
    try:
        client = HttpClient(
            host="localhost",
            port=8183,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        
        # Test creating a collection
        collection = client.create_collection(name="test_collection")
        print("Successfully connected to ChromaDB")
        
        # Cleanup
        client.delete_collection("test_collection")
        
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_chroma_connection()