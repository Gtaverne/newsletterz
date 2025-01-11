from typing import Dict, List, Optional
from datetime import datetime
import json
import os
import httpx
from chromadb import HttpClient, Documents, EmbeddingFunction
from chromadb.config import Settings
from src.email.gmail_fetcher import GmailFetcher

class OllamaEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str = "mxbai-embed-large", batch_size: int = 10):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=30.0)
        
    def __call__(self, texts: Documents) -> List[List[float]]:
        all_embeddings = []
        
        print(f"\nProcessing {len(texts)} texts in batches of {self.batch_size}")
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    response = self.client.post(
                        self.url,
                        json={"model": self.model_name, "prompt": text}
                    )
                    response_data = response.json()
                    # Change from 'embeddings' to 'embedding'
                    if "embedding" not in response_data:
                        print(f"Warning: No embedding in response: {response_data}")
                        continue
                        
                    embedding = response_data["embedding"]
                    if not embedding:
                        print(f"Warning: Empty embedding received")
                        continue
                        
                    batch_embeddings.append(embedding)
                    
                except Exception as e:
                    print(f"Error getting embedding: {e}")
                    continue
            
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch: {len(batch_embeddings)} embeddings generated")
            
        if all_embeddings:
            print(f"First embedding dimension: {len(all_embeddings[0])}")
            
        return all_embeddings

class EmailProcessor:
    def __init__(self, credentials_path: str, batch_size: int = 100):
        self.gmail = GmailFetcher(credentials_path)
        self.batch_size = batch_size
        self.chroma = HttpClient(
            host="localhost",
            port=8183,
            settings=Settings(chroma_client_auth_credentials="admin:admin")
        )
        self.failed_path = "data/failed_emails"
        os.makedirs(self.failed_path, exist_ok=True)
        
    def setup_collection(self):
        try:
            # Create embedder first
            self.embedder = OllamaEmbedding(batch_size=10)  # Smaller batch for testing
            
            # Test embedder
            print("\nTesting embedder...")
            test_embedding = self.embedder(["Test email content"])
            if not test_embedding:
                raise Exception("Embedder test failed - no embedding generated")
            print(f"Embedder test successful - dimension: {len(test_embedding[0])}")
            
            # Then create or get collection
            self.collection = self.chroma.get_or_create_collection(
                name="emails",
                metadata={"description": "Processed emails with embeddings"}
            )
            print("ChromaDB collection ready")
            
        except Exception as e:
            raise Exception(f"Failed to setup ChromaDB collection: {e}")

    def _save_failed_email(self, email: Dict, error: str):
        filename = f"{email['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.failed_path, filename)
        with open(filepath, 'w') as f:
            essential_data = {
                "id": email['id'],
                "thread_id": email['thread_id'],
                "date": email['internal_date'],
                "headers": email['headers'],
                "error": str(error)
            }
            json.dump(essential_data, f)
            print(f"Saved failed email to {filepath}")

    def process_emails(self, start_date: Optional[datetime] = None):
        page_token = None
        processed = 0
        
        print(f"\nStarting email processing from {start_date}")
        
        while True:
            try:
                result = self.gmail.fetch_emails(
                    start_date=start_date,
                    page_size=self.batch_size,
                    page_token=page_token
                )
                
                if not result['emails']:
                    print("No more emails to process")
                    break
                    
                self._process_batch(result['emails'])
                processed += len(result['emails'])
                print(f"\nTotal emails processed: {processed}")
                
                page_token = result.get('next_page_token')
                if not page_token:
                    print("No more pages to fetch")
                    break
                    
            except Exception as e:
                print(f"Failed to process batch: {e}")
                break

    def _process_batch(self, emails: List[Dict]):
        documents = []
        metadatas = []
        ids = []
        
        print(f"\nProcessing batch of {len(emails)} emails")
        
        for email in emails:
            try:
                # Skip if already exists
                if self._email_exists(email['id']):
                    print(f"Skipping existing email {email['id']}")
                    continue
                
                # Prepare data for batch insertion
                clean_text = email['clean_text']
                metadata = {
                    "email_id": email['id'],
                    "thread_id": email['thread_id'],
                    "date": email['internal_date'],
                    "subject": email['headers'].get('subject', ''),
                    "from": email['headers'].get('from', ''),
                    "to": email['headers'].get('to', ''),
                    "urls": json.dumps(email['urls']),
                    "has_urls": len(email['urls']) > 0,
                    "year_month": datetime.fromisoformat(email['internal_date']).strftime('%Y-%m')
                }
                
                documents.append(clean_text)
                metadatas.append(metadata)
                ids.append(email['id'])
                
            except Exception as e:
                print(f"Failed to process email {email['id']}: {e}")
                self._save_failed_email(email, str(e))
        
        # Batch insert into ChromaDB
        if documents:
            try:
                print(f"\nGenerating embeddings for {len(documents)} documents...")
                embeddings = self.embedder(documents)
                
                if not embeddings:
                    raise Exception("No embeddings generated")
                    
                print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
                
                print("Upserting to ChromaDB...")
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"Successfully upserted {len(documents)} documents to ChromaDB")
                
            except Exception as e:
                print(f"Failed to add batch to ChromaDB: {e}")
                for email, metadata in zip(emails, metadatas):
                    self._save_failed_email(email, str(e))

    def _email_exists(self, email_id: str) -> bool:
        try:
            results = self.collection.get(
                where={"email_id": email_id},
                limit=1
            )
            return len(results['ids']) > 0
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False

if __name__ == "__main__":
    processor = EmailProcessor("../secrets/credentials.json")
    processor.setup_collection()
    # Process emails since Jan 1st 2025
    start_date = datetime(2024, 7, 1)
    # processor.process_emails(start_date=start_date)
    processor.process_emails(start_date=start_date)