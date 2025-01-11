import httpx
import time

def wait_for_ollama(timeout=60):
    """Wait for Ollama to be ready, with timeout"""
    start_time = time.time()
    url = "http://localhost:11434/api/embeddings"
    
    while time.time() - start_time < timeout:
        try:
            # Try a simple embedding request
            response = httpx.post(
                url,
                json={"model": "llama3", "prompt": "test"},
                timeout=5.0
            )
            if response.status_code == 200:
                print("Ollama is ready!")
                print(f"Model response: {response.json()}")
                return True
        except Exception as e:
            print(f"Waiting for Ollama... ({str(e)})")
            time.sleep(2)
            
    print("Timeout waiting for Ollama")
    return False

if __name__ == "__main__":
    wait_for_ollama()