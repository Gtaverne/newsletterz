from typing import Dict, List, Optional
from datetime import datetime
import json
import os
import httpx
from chromadb import HttpClient, Documents, EmbeddingFunction
from chromadb.config import Settings

from src.email.gmail_fetcher import GmailFetcher
from src.search.company_registry import CompanyRegistry

class OllamaEmbedding(EmbeddingFunction):
    def __init__(self, model_name: str = "mxbai-embed-large", batch_size: int = 50):
        self.model_name = model_name
        self.url = "http://localhost:11434/api/embeddings"
        self.batch_size = batch_size
        self.client = httpx.Client(timeout=60.0)  # Increased timeout for larger batches
        
    def __call__(self, texts: Documents) -> List[List[float]]:
        all_embeddings = []
        
        print(f"\nProcessing {len(texts)} texts in batches of {self.batch_size}")
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    response = self.client.post(
                        self.url,
                        json={"model": self.model_name, "prompt": text}
                    )
                    embedding = response.json().get("embedding")
                    if embedding:
                        batch_embeddings.append(embedding)
                    else:
                        print(f"Warning: No embedding in response")
                        
                except Exception as e:
                    print(f"Error getting embedding: {e}")
                    continue
            
            all_embeddings.extend(batch_embeddings)
            print(f"Processed batch: {len(batch_embeddings)} embeddings generated")
            
        return all_embeddings

def chunk_email(subject: str, content: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """
    Split email content into overlapping chunks with context
    """
    chunks = []
    prefix = f"Subject: {subject}\n\n"
    prefix_len = len(prefix)
    
    content_chunk_size = chunk_size - prefix_len
    start = 0
    position = 0
    
    while start < len(content):
        chunk_end = start + content_chunk_size
        if chunk_end < len(content):
            breakpoint = content.rfind('\n', start, chunk_end)
            if breakpoint > start:
                chunk_end = breakpoint
                
        chunk_text = prefix + content[start:chunk_end]
        chunks.append({
            'text': chunk_text,
            'position': position,
            'total_chunks': None
        })
        
        start = chunk_end - overlap
        position += 1
    
    total = len(chunks)
    for chunk in chunks:
        chunk['total_chunks'] = total
        
    return chunks

def analyze_email_length(email: Dict) -> Dict:
    """Analyze email content length and structure"""
    subject = email['headers'].get('subject', '')
    content = email['clean_text']
    
    return {
        'id': email['id'],
        'char_length': len(content),
        'subject_length': len(subject),
        'has_content': bool(content.strip()),
        'approx_tokens': len(content.split())  # Rough approximation
    }

class EmailProcessor:
    def __init__(self, credentials_path: str, batch_size: int = 200):  
        self.gmail = GmailFetcher(credentials_path)
        self.batch_size = batch_size
        self.chroma = HttpClient(
            host="localhost",
            port=8183,
            settings=Settings(chroma_client_auth_credentials="admin:admin")
        )
        self.failed_path = "data/failed_emails"
        os.makedirs(self.failed_path, exist_ok=True)
        
    def _save_failed_email(self, email: Dict, error: str, error_type: str = "processing"):
        """Enhanced error logging"""
        filename = f"{email['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.failed_path, filename)
        
        try:
            error_data = {
                "email_id": email['id'],
                "thread_id": email['thread_id'],
                "date": email['internal_date'],
                "headers": email['headers'],
                "error": str(error),
                "error_type": error_type,
                "content_stats": analyze_email_length(email)
            }
            
            with open(filepath, 'w') as f:
                json.dump(error_data, f, indent=2)
            print(f"Saved error log to {filepath}")
        except Exception as e:
            print(f"Failed to save error log: {e}")

    def _process_email(self, email: Dict) -> Optional[Dict]:
        """Process a single email, returning metadata if successful"""
        try:
            # Basic validation
            if not email.get('clean_text'):
                print(f"Skipping email {email['id']} - no content")
                return None
                
            date_obj = datetime.fromisoformat(email['internal_date'])
            from_field = email['headers'].get('from', '')
            company = CompanyRegistry.match_sender(from_field)
            
            # Create metadata
            metadata = {
                "email_id": email['id'],
                "thread_id": email['thread_id'],
                "date": int(date_obj.timestamp()),  # Convert to Unix timestamp
                "subject": email['headers'].get('subject', 'No Subject'),
                "from": from_field,
                "company": company,
                "content_length": len(email['clean_text'])
            }
            
            return metadata
        except Exception as e:
            self._save_failed_email(email, str(e), "processing")
            return None

    def _process_batch(self, emails: List[Dict]):
        """Process a batch of emails with improved error handling"""
        documents = []
        metadatas = []
        ids = []
        
        print(f"\nProcessing batch of {len(emails)} emails")
        
        # First pass - analyze all emails
        analysis = [analyze_email_length(email) for email in emails]
        print("\nContent analysis:")
        print(f"Average content length: {sum(a['char_length'] for a in analysis) / len(analysis):.0f} chars")
        print(f"Max content length: {max(a['char_length'] for a in analysis)} chars")
        
        for email in emails:
            try:
                metadata = self._process_email(email)
                if metadata:
                    documents.append(email['clean_text'])
                    metadatas.append(metadata)
                    ids.append(email['id'])
            except Exception as e:
                self._save_failed_email(email, str(e), "batch_processing")
        
        if documents:
            try:
                print(f"\nGenerating embeddings for {len(documents)} emails...")
                embeddings = self.embedder(documents)
                
                if not (len(ids) == len(metadatas) == len(embeddings) == len(documents)):
                    raise ValueError(
                        f"Length mismatch: ids={len(ids)}, metadatas={len(metadatas)}, "
                        f"embeddings={len(embeddings)}, documents={len(documents)}"
                    )
                
                self.collection.upsert(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"Successfully processed {len(documents)} emails")
                
            except Exception as e:
                print(f"Batch processing failed: {e}")
                for email in emails:
                    self._save_failed_email(email, str(e), "embedding_failed")


    def setup_collection(self):
        """Create embedder and collection"""
        try:
            # Create embedder first
            self.embedder = OllamaEmbedding(batch_size=50)
            
            # Test embedder
            print("\nTesting embedder...")
            test_embedding = self.embedder(["Test email content"])
            if not test_embedding:
                raise Exception("Embedder test failed - no embedding generated")
            print(f"Embedder test successful - dimension: {len(test_embedding[0])}")
            
            # Create collection if it doesn't exist
            try:
                self.collection = self.chroma.get_collection("emails")
                print("Using existing ChromaDB collection")
            except Exception:
                self.collection = self.chroma.create_collection(
                    name="emails",
                    metadata={"description": "Processed emails with embeddings"}
                )
                print("Created new ChromaDB collection")
            
        except Exception as e:
            raise Exception(f"Failed to setup ChromaDB collection: {e}")

    def reset_database(self):
        """Drop and recreate the emails collection"""
        try:
            print("\nDropping database...")
            try:
                self.chroma.delete_collection("emails")
                print("Deleted existing collection")
            except Exception:
                print("No existing collection to delete")
                
            self.setup_collection()
            print("Database reset completed")
            
        except Exception as e:
            raise Exception(f"Database reset failed: {e}")

    def get_processed_email_ids(self) -> set:
        """Get set of already processed email IDs"""
        try:
            results = self.collection.get(
                include=['metadatas'],
                where={"chunk_position": 0}  # Only get first chunks to avoid duplicates
            )
            return {m['email_id'] for m in results['metadatas']}
        except Exception:
            return set()  # Return empty set if collection doesn't exist yet

    def process_emails(self, start_date: Optional[datetime] = None):
        """Process emails with progress tracking and deduplication"""
        try:
            self.setup_collection()
            processed_ids = self.get_processed_email_ids()
            
            page_token = None
            processed_emails = 0
            processed_chunks = 0
            
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
                    
                    print(f"\nReceived batch of {len(result['emails'])} emails")
                    
                    # Filter out already processed emails
                    new_emails = [
                        email for email in result['emails'] 
                        if email['id'] not in processed_ids
                    ]
                    
                    print(f"Found {len(new_emails)} new emails to process")
                    print(f"Skipping {len(result['emails']) - len(new_emails)} already processed emails")
                    
                    if new_emails:
                        self._process_batch(new_emails)
                        processed_emails += len(new_emails)
                        # Update processed IDs
                        processed_ids.update(email['id'] for email in new_emails)
                    else:
                        print("No new emails in this batch")
                    
                    # Get total chunks from collection
                    current_chunks = len(self.collection.get()['ids'])
                    chunks_in_batch = current_chunks - processed_chunks
                    processed_chunks = current_chunks
                    
                    print(f"\nProgress:")
                    print(f"- Emails processed: {processed_emails}")
                    print(f"- Total chunks: {processed_chunks}")
                    if processed_emails > 0:
                        print(f"- Average chunks per email: {processed_chunks/processed_emails:.1f}")
                    
                    page_token = result.get('next_page_token')
                    if not page_token:
                        print("No more pages to fetch")
                        break
                        
                except Exception as e:
                    print(f"Failed to process batch: {e}")
                    break
                    
            print(f"\nProcessing completed:")
            print(f"- Total emails: {processed_emails}")
            print(f"- Total chunks: {processed_chunks}")
            if processed_emails > 0:
                print(f"- Average chunks per email: {processed_chunks/processed_emails:.1f}")
            else:
                print("- No emails were processed")
            
        except Exception as e:
            raise Exception(f"Email processing failed: {e}")

if __name__ == "__main__":
    # Example usage
    processor = EmailProcessor("../secrets/credentials.json")
    # processor.reset_database()

    start_date = datetime(2024, 1, 1)  # Process emails from 2023
    processor.process_emails(start_date=start_date)