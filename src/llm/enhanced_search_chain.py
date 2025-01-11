from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Optional
from datetime import datetime
import httpx
import json
from chromadb import HttpClient
from chromadb.config import Settings

class EnhancedEmailSearchChain:
    # Define the template as a class attribute
    QUERY_TEMPLATE = """You are an AI assistant specialized in analyzing email search queries and extracting structured information.

Analyze this search query: "{query}"

Think through step by step:
1. What is the primary intent of this query?
   - Is it asking for a count? ("how many...")
   - Is it asking for the most recent? ("latest...", "most recent...")
   - Is it asking for a timeline or trend?
   - Is it a general search?

2. What is the main topic or subject matter?
   - What specific topics or keywords are mentioned?
   - Are there any related concepts we should consider?

3. Are there any source or sender constraints?
   - Is it asking for emails from specific companies?
   - Are there domain or sender restrictions?

4. Are there any time constraints?
   - Any specific dates or date ranges?
   - Any relative time periods? ("last month", "recent", etc.)

5. What type of response would best answer this query?
   - A simple count?
   - A chronological list?
   - A summary of findings?
   - A detailed analysis?

Based on your analysis, provide a structured response in this JSON format:
{{
    "intent": "count" or "latest" or "search" or "timeline",
    "topic": "the main topic or subject",
    "filters": {{
        "company": "company name if mentioned",
        "date_range": "any date constraints",
        "keywords": ["relevant", "search", "terms"]
    }},
    "response_type": "count" or "list" or "summary",
    "reasoning": "brief explanation of your analysis"
}}

Remember to be thorough in your analysis and ensure the JSON structure accurately captures the query intent."""

    def __init__(self, host: str = "localhost", port: int = 8183):
        """Initialize search chain with ChromaDB and Ollama"""
        # Initialize ChromaDB
        self.chroma = HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        self.collection = self.chroma.get_collection("emails")
        
        # Initialize Ollama
        self.llm = OllamaLLM(
            model="qwen2.5-coder:32b",
            base_url="http://localhost:11434",
            temperature=0.1  # Lower temperature for more consistent parsing
        )
        self.prompt = ChatPromptTemplate.from_template(self.QUERY_TEMPLATE)
        self.embeddings_url = "http://localhost:11434/api/embeddings"

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding using mxbai-embed-large model"""
        response = httpx.post(
            self.embeddings_url,
            json={"model": "mxbai-embed-large", "prompt": text},
            timeout=30.0
        )
        return response.json()["embedding"]

    async def _parse_query(self, query: str) -> Dict:
        """Parse natural language query into structured components"""
        try:
            print("\nAnalyzing query with Qwen...")
            chain = self.prompt | self.llm
            result = await chain.ainvoke({"query": query})
            print("\nQwen Analysis:")
            print(result)
            
            # Extract JSON from response if needed
            if isinstance(result, str):
                # Find JSON content between curly braces
                start = result.find('{')
                end = result.rfind('}') + 1
                if start >= 0 and end > start:
                    result = result[start:end]
                
            parsed = json.loads(result)
            print("\nParsed structure:", json.dumps(parsed, indent=2))
            return parsed
            
        except Exception as e:
            print(f"Error parsing query: {e}")
            # Fallback structure
            return {
                "intent": "search",
                "topic": "cloud computing",
                "filters": {
                    "company": "McKinsey",
                    "keywords": ["cloud"]
                },
                "response_type": "list"
            }

    def _build_where_filter(self, parsed_query: Dict) -> Optional[Dict]:
        """Build ChromaDB where filter from parsed query"""
        conditions = {}
        
        # Handle company/domain filtering
        if company := parsed_query.get('filters', {}).get('company'):
            company = company.lower()
            company_domains = {
                'mckinsey': ['mckinsey.com', 'email.mckinsey.com'],
                'deloitte': ['deloitte.com', 'email.deloitte.com'],
                'bcg': ['bcg.com', 'email.bcg.com']
            }
            domains = company_domains.get(company, [f"{company}.com"])
            # Using $in operator which is supported by ChromaDB
            conditions["from"] = {"$in": [
                domain for d in domains 
                for domain in [f"@{d}", f"<{d}>", f"{d}"]
            ]}
                
        return conditions if conditions else None

    def _format_results(self, results: Dict, parsed_query: Dict) -> Dict:
        """Format results based on query intent"""
        if not results.get('ids'):
            return {"type": "error", "message": "No results found"}
            
        formatted = []
        for i in range(len(results['ids'][0])):
            item = {
                'id': results['ids'][0][i],
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i]
            }
            
            if parsed_query['intent'] == 'count':
                return {
                    'type': 'count',
                    'count': len(results['ids'][0]),
                    'topic': parsed_query['topic'],
                    'company': parsed_query.get('filters', {}).get('company')
                }
            
            elif parsed_query['intent'] == 'timeline':
                formatted.append({
                    'date': item['metadata']['date'],
                    'subject': item['metadata']['subject'],
                    'from': item['metadata']['from'],
                    'preview': item['content'][:200] + '...' if len(item['content']) > 200 else item['content']
                })
                formatted.sort(key=lambda x: x['date'])
            
            else:  # Default search
                formatted.append({
                    'subject': item['metadata']['subject'],
                    'from': item['metadata']['from'],
                    'date': item['metadata']['date'],
                    'preview': item['content'][:200] + '...' if len(item['content']) > 200 else item['content']
                })
                
        return {
            'type': parsed_query['intent'],
            'results': formatted,
            'count': len(formatted),
            'query_analysis': parsed_query
        }
    
    async def search(self, query: str, limit: int = 1000, min_relevance: float = 0.7) -> Dict:
        """
        Process natural language query and return structured results with proper AND handling
        """
        try:
            # Parse query intent
            parsed = await self._parse_query(query)
            print("\nParsed query:", json.dumps(parsed, indent=2))
            
            keywords = parsed.get('filters', {}).get('keywords', [])
            
            if len(keywords) > 1:
                # For multiple keywords, get separate embeddings and results
                all_results = []
                for keyword in keywords:
                    search_text = f"{keyword}"
                    print(f"\nGenerating embedding for: '{search_text}'")
                    query_embedding = self.get_embedding(search_text)
                    
                    # Get results for each keyword
                    results = self.collection.query(
                        query_embeddings=[query_embedding],
                        n_results=min(limit * 2, 10000),
                        where=self._build_where_filter(parsed),
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Filter by relevance
                    filtered = self._filter_by_relevance(results, min_relevance)
                    all_results.append(set(filtered['ids'][0]))
                
                # Find intersection of all result sets
                common_ids = set.intersection(*all_results)
                print(f"\nFound {len(common_ids)} documents matching ALL keywords")
                
                # Get full details for intersecting results
                if common_ids:
                    final_results = self.collection.get(
                        ids=list(common_ids),
                        include=['documents', 'metadatas']
                    )
                    
                    # For count queries, return simplified format
                    if parsed['intent'] == 'count':
                        return {
                            "type": "count",
                            "count": len(common_ids),
                            "topic": parsed['topic'],
                            "keywords": keywords,
                            "individual_counts": {
                                kw: len(results) for kw, results in zip(keywords, all_results)
                            }
                        }
                    else:
                        # For other queries, get full details
                        final_results = self.collection.get(
                            ids=list(common_ids),
                            include=['documents', 'metadatas']
                        )
                        final_results['distances'] = [[1.0] * len(common_ids)]
                        return self._format_results(final_results, parsed)
                else:
                    return {
                        "type": "count",
                        "count": 0,
                        "topic": parsed['topic'],
                        "min_relevance": min_relevance
                    }
                    
            else:
                # Single keyword search - use original approach
                search_text = parsed['topic']
                print(f"\nGenerating embedding for search text: '{search_text}'")
                query_embedding = self.get_embedding(search_text)
                
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(limit * 2, 10000),
                    where=self._build_where_filter(parsed),
                    include=['documents', 'metadatas', 'distances']
                )
                
                filtered_results = self._filter_by_relevance(results, min_relevance)
                return self._format_results(filtered_results, parsed)
                
        except Exception as e:
            print(f"Search error: {e}")
            return {'type': 'error', 'message': str(e)}

    def _filter_by_relevance(self, results: Dict, min_relevance: float) -> Dict:
        """Filter results based on normalized relevance score"""
        if not results['ids'][0]:
            return results
            
        distances = results['distances'][0]
        min_dist = min(distances)
        max_dist = max(distances)
        dist_range = max_dist - min_dist
        
        filtered_indices = []
        relevance_scores = []
        
        for i, distance in enumerate(distances):
            # Normalize to 0-1 range (1 being most relevant)
            relevance = 1 - ((distance - min_dist) / dist_range if dist_range > 0 else 0)
            
            if relevance >= min_relevance:
                filtered_indices.append(i)
                relevance_scores.append(relevance)
        
        # Create filtered results
        filtered = {
            'ids': [[results['ids'][0][i] for i in filtered_indices]],
            'documents': [[results['documents'][0][i] for i in filtered_indices]],
            'metadatas': [[results['metadatas'][0][i] for i in filtered_indices]],
            'distances': [relevance_scores]
        }
        
        print(f"\nFiltered {len(results['ids'][0])} results to {len(filtered_indices)} based on relevance threshold {min_relevance}")
        return filtered

    def _format_results(self, results: Dict, parsed_query: Dict) -> Dict:
        """Format results based on query intent"""
        if not results.get('ids', [[]])[0]:
            return {"type": "error", "message": "No results found"}
            
        formatted = []
        
        for i in range(len(results['ids'][0])):
            relevance = 1 / (1 + abs(results['distances'][0][i]))
            item = {
                'relevance': round(relevance, 3),
                'subject': results['metadatas'][0][i]['subject'],
                'from': results['metadatas'][0][i]['from'],
                'date': results['metadatas'][0][i]['date'],
                'preview': results['documents'][0][i][:200] + '...' if len(results['documents'][0][i]) > 200 else results['documents'][0][i]
            }
            
            if parsed_query['intent'] == 'count':
                return {
                    'type': 'count',
                    'count': len(results['ids'][0]),
                    'topic': parsed_query['topic'],
                    'min_relevance': round(min(results['distances'][0]), 3),
                    'max_relevance': round(max(results['distances'][0]), 3),
                    'company': parsed_query.get('filters', {}).get('company')
                }
                
            formatted.append(item)
            
        # Sort by relevance for non-count queries
        if parsed_query['intent'] != 'count':
            formatted.sort(key=lambda x: x['relevance'], reverse=True)
            
        return {
            'type': parsed_query['intent'],
            'results': formatted,
            'count': len(formatted),
            'query_analysis': parsed_query
        }