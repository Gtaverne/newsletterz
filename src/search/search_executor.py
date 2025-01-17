from typing import Dict, List, Optional, Any
from datetime import datetime
import httpx
from chromadb import HttpClient
from chromadb.config import Settings
from .models import QueryIntent, SearchResponse, EmailReference
from .company_registry import CompanyRegistry

class SearchExecutor:
    def __init__(self, host: str = "localhost", port: int = 8183, verbose: bool = False):
        self.chroma = HttpClient(
            host=host,
            port=port,
            settings=Settings(
                chroma_client_auth_credentials="admin:admin"
            )
        )
        self.collection = self.chroma.get_collection("emails")
        self.embeddings_url = "http://localhost:11434/api/embeddings"
        self.verbose = verbose

    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using mxbai-embed-large model"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.embeddings_url,
                json={"model": "mxbai-embed-large", "prompt": text},
                timeout=30.0
            )
            return response.json()["embedding"]

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
            
        filters = []
        if start := time_range.get('start'):
            # Convert to Unix timestamp (seconds since epoch)
            timestamp = int(start.timestamp())
            filters.append({"date": {"$gte": timestamp}})
        if end := time_range.get('end'):
            timestamp = int(end.timestamp())
            filters.append({"date": {"$lte": timestamp}})
                
        # If we have both start and end, combine them with $and
        if len(filters) > 1:
            return {"$and": filters}
        # If we have only one filter, return it directly
        return filters[0] if filters else None


    def _combine_filters(self, filters: List[Dict[str, Any]]) -> Optional[Dict]:
        """Combine multiple filters with AND logic"""
        valid_filters = [f for f in filters if f is not None]
        if not valid_filters:
            return None
        if len(valid_filters) == 1:
            return valid_filters[0]
        return {"$and": valid_filters}


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
                        limit: int = 20) -> Dict:
        """Format raw ChromaDB results based on query intent"""
        if not chroma_results['ids'][0]:
            return {
                "type": "empty",
                "message": "No results found matching the criteria",
                "query_info": intent.model_dump()
            }

        total_results = len(chroma_results['ids'][0])
        
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
                "distance": round(distance, 3),
                "content": chroma_results['documents'][0][i]
            }
            formatted_results.append(result)

        # Sort based on query type
        if intent.type != "timeline":
            formatted_results.sort(key=lambda x: x['distance'])
        else:
            formatted_results.sort(key=lambda x: x['date'])

        return {
            "type": intent.type,
            "total_results": total_results,
            "returned_results": len(formatted_results),
            "results": formatted_results,
            "query_info": intent.model_dump()
        }


    def _normalize_distances(self, distances: List[float]) -> List[float]:
        """
        Normalize distances to similarity scores (0-1 range)
        Higher score = more similar
        """
        if not distances:
            return []
            
        # Typical mxbai-embed-large distances are in range 150-300
        # Let's use a more reasonable normalization
        max_expected_distance = 300
        
        # Convert to similarity scores
        similarities = [
            max(0, 1 - (distance / max_expected_distance))
            for distance in distances
        ]
        
        return similarities
    
    def _build_semantic_query(self, topic: str, semantic_context: dict) -> str:
        """Build rich semantic query from topic and context"""
        query_parts = [topic]
        
        # Add core concepts if available
        if concepts := semantic_context.get('core_concepts', []):
            query_parts.append(f"Related to: {', '.join(concepts)}")
            
        # Add main aspects if available
        if aspects := semantic_context.get('aspects', []):
            query_parts.append(f"Including aspects: {', '.join(aspects)}")
        
        return " ".join(query_parts)

    def _consolidate_results(self, results: List[Dict], similarity_threshold: float = 0.95) -> List[Dict]:
        """
        Consolidate similar results and remove duplicates
        
        Args:
            results: List of search results
            similarity_threshold: Threshold for considering results similar (0-1)
        """
        consolidated = []
        seen_threads = set()
        
        # Sort by date first to prioritize recent content
        sorted_results = sorted(results, key=lambda x: x['metadata']['date'], reverse=True)
        
        for result in sorted_results:
            thread_id = result['metadata'].get('thread_id')
            
            # If part of a seen thread, skip
            if thread_id and thread_id in seen_threads:
                continue
                
            # Check similarity with existing consolidated results
            is_similar = False
            for existing in consolidated:
                # If similarity scores are very close, consider them similar content
                score_diff = abs(existing['similarity'] - result['similarity'])
                if score_diff < (1 - similarity_threshold):
                    is_similar = True
                    # Keep the one with better similarity score
                    if result['similarity'] > existing['similarity']:
                        consolidated.remove(existing)
                        consolidated.append(result)
                    break
            
            if not is_similar:
                consolidated.append(result)
            
            if thread_id:
                seen_threads.add(thread_id)
        
        return consolidated

    async def execute_search(self, intent: QueryIntent, limit: int = 1000) -> Dict:
        """Execute a search based on parsed intent with relevance threshold"""
        company_filter = self._build_company_filter(intent.filters.companies)
        date_filter = self._build_date_filter(intent.filters.time_range.model_dump() 
                                            if intent.filters.time_range else None)
        where_filter = self._combine_filters([company_filter, date_filter])

        semantic_query = self._build_semantic_query(
            intent.topic,
            getattr(intent, 'semantic_context', {})
        )
        
        if self.verbose:
            print(f"\nExecuting search: {semantic_query}")
            if where_filter:
                print(f"Filters: {where_filter}")

        query_embedding = await self._get_embedding(semantic_query)
        
        # Initial search with larger limit to allow for filtering
        initial_limit = min(limit * 2, 4000)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=initial_limit,
            where=where_filter,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Early exit if no results
        if not results['ids'][0]:
            return {
                "type": "empty",
                "message": "No results found matching the criteria",
                "query_info": intent.model_dump()
            }

        distances = results['distances'][0]
        
        # Find relevance cliff
        min_distance = min(distances)
        threshold = min_distance * 2.5  # Results with distance > 2.5x the best match are considered irrelevant
        
        # Filter results by threshold
        relevant_indices = []
        for i, distance in enumerate(distances):
            if distance <= threshold:
                relevant_indices.append(i)
            else:
                # Stop when we hit irrelevant results
                break
                
        if self.verbose:
            print(f"\nRelevance filtering:")
            print(f"Best match distance: {min_distance:.3f}")
            print(f"Threshold: {threshold:.3f}")
            print(f"Relevant results: {len(relevant_indices)}/{len(distances)}")

        # Build final results
        filtered_results = []
        for i in relevant_indices:
            metadata = self._format_email_metadata(results['metadatas'][0][i])
            filtered_results.append({
                "id": results['ids'][0][i],
                "subject": metadata.get('subject', 'No subject'),
                "from": metadata.get('from', 'Unknown sender'),
                "company": metadata.get('company', 'unknown'),
                "date": metadata.get('date'),
                "content": results['documents'][0][i],
                "distance": distances[i]
            })

        return {
            "type": intent.type,
            "total_results": len(filtered_results),
            "results": filtered_results,
            "query_info": intent.model_dump(),
            "threshold_info": {
                "min_distance": min_distance,
                "threshold": threshold,
                "total_retrieved": len(distances)
            }
        }
