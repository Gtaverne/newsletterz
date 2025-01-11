from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from typing import List, Dict, Optional
from datetime import datetime
import json
from chromadb import HttpClient
from chromadb.config import Settings

class EmailSearchChain:
    def __init__(self, host: str = "localhost", port: int = 8183):
        """Initialize with ChromaDB connection"""
        self.chroma = HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        self.collection = self.chroma.get_collection("emails")
        self.llm = Ollama(model="llama3")
        
        # Chain for parsing query intent
        self.query_parser = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["query"],
                template="""Parse this email search query into components.
                Query: {query}
                
                Rules:
                - Extract time periods if mentioned (this year, last month, since 2024, etc)
                - Extract sender/domain filters if mentioned
                - Identify if user wants a list, summary, or timeline
                - Extract the main topic or search terms
                
                Output JSON with these fields:
                - main_topic: main subject to search for
                - time_filter: any time constraints as ISO dates
                - source_filter: any sender/domain filters
                - query_type: either 'list', 'summarize', or 'timeline'
                """
            )
        )
    
    def _parse_query(self, query: str) -> Dict:
        """Parse natural language query into structured components"""
        try:
            result = self.query_parser.run(query)
            return json.loads(result)
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback to basic search
            return {
                "main_topic": query,
                "time_filter": None,
                "source_filter": None,
                "query_type": "list"
            }
    
    def _build_where_filter(self, parsed_query: Dict) -> Optional[Dict]:
        """Build ChromaDB where filter from parsed query"""
        conditions = {}
        
        # Add source filter if specified
        if source := parsed_query.get('source_filter'):
            conditions["from"] = {"$eq": f"*{source}*"}
            
        # Add time filter if specified
        if time_filter := parsed_query.get('time_filter'):
            try:
                # Assuming time_filter is an ISO date string
                date_str = datetime.fromisoformat(time_filter).strftime('%Y-%m')
                conditions["year_month"] = {"$gte": date_str}
            except:
                pass
                
        return conditions if conditions else None
    
    def _format_results(self, results: Dict, query_type: str = "list") -> Dict:
        """Format ChromaDB results based on query type"""
        formatted = []
        
        for i in range(len(results['ids'][0])):
            item = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            
            if query_type == "list":
                formatted.append({
                    'subject': item['metadata']['subject'],
                    'from': item['metadata']['from'],
                    'date': item['metadata']['date'],
                    'preview': item['content'][:200] + '...' if item['content'] else 'No content'
                })
            else:
                formatted.append(item)
                
        return {
            'type': query_type,
            'results': formatted,
            'count': len(formatted)
        }
    
    def search(self, query: str, limit: int = 10) -> Dict:
        """Process natural language query and return structured results"""
        # Parse query intent
        parsed = self._parse_query(query)
        print(f"Parsed query: {parsed}")
        
        # Build filter
        where_filter = self._build_where_filter(parsed)
        
        try:
            # Execute search
            results = self.collection.query(
                query_texts=[parsed['main_topic']],
                n_results=limit,
                where=where_filter,
                include=['documents', 'metadatas']
            )
            
            # Format results
            return self._format_results(results, parsed['query_type'])
            
        except Exception as e:
            print(f"Search error: {e}")
            return {'type': 'error', 'message': str(e)}