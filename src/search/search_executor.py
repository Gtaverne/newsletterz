from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx
from chromadb import HttpClient
from chromadb.config import Settings
from .models import QueryIntent, SearchResponse, EmailReference
from .company_registry import CompanyRegistry

class SearchExecutor:
    def __init__(self, host: str = "localhost", port: int = 8183):
        """Initialize search executor with ChromaDB connection"""
        self.chroma = HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        self.collection = self.chroma.get_collection("emails")
        self.embeddings_url = "http://localhost:11434/api/embeddings"
        
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using mxbai-embed-large model"""
        try:
            response = httpx.post(
                self.embeddings_url,
                json={"model": "mxbai-embed-large", "prompt": text},
                timeout=30.0
            )
            return response.json()["embedding"]
        except Exception as e:
            raise Exception(f"Failed to get embedding: {str(e)}")

    def _build_company_filter(self, companies: List[str]) -> Optional[Dict]:
        """
        Build ChromaDB filter for company matching.
        Uses CompanyRegistry to handle company name variations.
        """
        if not companies:
            return None

        # Convert company names to registry keys and variations
        company_variations = []
        for company in companies:
            company = company.lower()
            # If it's a direct match in registry
            if company in CompanyRegistry.COMPANIES:
                company_variations.append(company)
            # Check if it matches any variations
            for reg_company, variations in CompanyRegistry.COMPANIES.items():
                if company in [v.lower() for v in variations]:
                    company_variations.append(reg_company)
        
        if not company_variations:
            return None

        # Use $in operator for multiple companies
        return {
            "company": {"$in": company_variations}
        }

    def _build_date_filter(self, time_range: Optional[Dict]) -> Optional[Dict]:
        """Build ChromaDB filter for date range"""
        if not time_range or (not time_range.get('start') and not time_range.get('end')):
            return None
            
        date_conditions = {}
        if start := time_range.get('start'):
            date_conditions["$gte"] = start
        if end := time_range.get('end'):
            date_conditions["$lte"] = end
            
        return {"date": date_conditions} if date_conditions else None

    def _combine_filters(self, filters: List[Dict[str, Any]]) -> Optional[Dict]:
        """Combine multiple filters with AND logic"""
        valid_filters = [f for f in filters if f is not None]
        if not valid_filters:
            return None
        if len(valid_filters) == 1:
            return valid_filters[0]
        return {"$and": valid_filters}

    def _calculate_relevance(self, distance: float) -> float:
        """Convert distance to relevance score (0-1)"""
        return 1 / (1 + abs(distance))

    def _format_email_metadata(self, metadata: Dict) -> Dict:
        """Format email metadata with consistent company information"""
        formatted = metadata.copy()
        # Ensure company information is present
        if 'from' in formatted:
            company = formatted.get('company', 'unknown')
            if company == 'unknown':
                # Try to match company again if needed
                company = CompanyRegistry.match_sender(formatted['from'])
            formatted['company'] = company
        return formatted

    def _format_results(self, 
                       chroma_results: Dict, 
                       intent: QueryIntent,
                       limit: int = 10) -> Dict:
        """Format raw ChromaDB results based on query intent"""
        if not chroma_results['ids'][0]:
            return {
                "type": "empty",
                "message": "No results found matching the criteria",
                "query_info": intent.model_dump()
            }

        total_results = len(chroma_results['ids'][0])
        
        # For count queries, return early with just the count
        if intent.type == "count":
            return {
                "type": "count",
                "count": total_results,
                "query_info": intent.model_dump()
            }

        # Process and format results
        formatted_results = []
        for i in range(min(total_results, limit)):
            metadata = self._format_email_metadata(chroma_results['metadatas'][0][i])
            distance = chroma_results['distances'][0][i]
            
            result = {
                "id": chroma_results['ids'][0][i],
                "subject": metadata.get('subject', 'No subject'),
                "from": metadata.get('from', 'Unknown sender'),
                "company": metadata.get('company', 'unknown'),
                "date": metadata.get('date'),
                "relevance": self._calculate_relevance(distance),
                "content": chroma_results['documents'][0][i]
            }
            formatted_results.append(result)

        # Sort by relevance for all query types except timeline
        if intent.type != "timeline":
            formatted_results.sort(key=lambda x: x['relevance'], reverse=True)
        else:
            # For timeline queries, sort by date
            formatted_results.sort(key=lambda x: x['date'])

        return {
            "type": intent.type,
            "total_results": total_results,
            "returned_results": len(formatted_results),
            "results": formatted_results,
            "query_info": intent.model_dump()
        }

    async def execute_search(self, 
                           intent: QueryIntent, 
                           limit: int = 1000,
                           min_relevance: float = 0.7) -> Dict:
        """Execute search based on parsed intent"""
        try:
            # Build filters
            company_filter = self._build_company_filter(intent.filters.companies)
            date_filter = self._build_date_filter(intent.filters.time_range.model_dump() 
                                                if intent.filters.time_range else None)
            
            # Debug print filters
            print("\nSearch filters:")
            print(f"Company filter: {company_filter}")
            print(f"Date filter: {date_filter}")
            
            # Combine all filters
            where_filter = self._combine_filters([company_filter, date_filter])
            
            # Prepare search text combining topic and keywords
            search_text = intent.topic
            if intent.filters.keywords:
                search_text += " " + " ".join(intent.filters.keywords)
            
            print(f"\nSearch text: {search_text}")
            
            # Get embedding for search
            query_embedding = self._get_embedding(search_text)
            
            # Execute search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 2, 10000),  # Get extra results for filtering
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            print(f"\nRaw results count: {len(results['ids'][0])}")
            
            # Format results based on intent
            return self._format_results(results, intent)
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return {
                "type": "error",
                "message": str(e),
                "query_info": intent.model_dump()
            }