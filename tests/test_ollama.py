import httpx

def test_ollama():
    url = "http://localhost:11434/api/embeddings"
    text = "This is a test"
    
    try:
        response = httpx.post(
            url,
            json={"model": "mxbai-embed-large", "prompt": text}
        )
        result = response.json()
        print("Embedding length:", len(result["embedding"]))
        print("First few values:", result["embedding"][:5])
        return True
    except Exception as e:
        print(f"Ollama test failed: {e}")
        return False

if __name__ == "__main__":
    test_ollama()