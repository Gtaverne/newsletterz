from langchain_ollama import OllamaLLM
from typing import Optional
import json
from datetime import datetime
from .models import QueryIntent, FilterConfig, TimeRange
from .company_registry import CompanyRegistry

QUERY_TEMPLATE = """You are an AI assistant specialized in analyzing email search queries.
Available companies: {companies}

Analyze this search query: "{query}"

Think through step by step:
1. What is the primary intent?
   - Is it asking for a count? ("how many...")
   - Is it asking for a summary/trends? ("what are the trends...")
   - Is it asking for a list? ("show me...")

2. What is the main topic and its semantic context?
   - Extract the core subject matter
   - Expand the topic with related concepts and terminology that would be relevant in a business context
   - Include industry-specific terms and jargon
   - Consider different aspects and subdomains of the topic

3. Identify companies:
   - Look for explicit company names
   - Understand business terms and translations
   - Map industry groups to individual companies
   - Only use companies from our available list

Output a JSON object that matches this schema:
{{
    "type": "count" | "summary" | "trend" | "list",
    "topic": "main topic or subject",
    "semantic_context": {{
        "core_concepts": ["list of core concepts related to the topic"],
        "related_terms": ["list of related business/industry terms"],
        "aspects": ["different aspects or angles of the topic"]
    }},
    "filters": {{
        "companies": ["list of known companies that match the query"],
        "time_range": {{
            "start": "ISO date or null",
            "end": "ISO date or null",
            "description": "human readable description"
        }},
        "keywords": ["relevant keywords"]
    }},
    "reasoning": "explanation of analysis"
}}

EXAMPLE:
For a query about "supply chain trends":
{{
    "type": "trend",
    "topic": "supply chain trends",
    "semantic_context": {{
        "core_concepts": ["supply chain management", "logistics", "procurement"],
        "related_terms": ["inventory optimization", "warehousing", "distribution networks", "last-mile delivery"],
        "aspects": ["risk management", "sustainability", "digital transformation", "resilience"]
    }},
    ...
}}"""

class IntentParser:
    def __init__(self, model_name: str = "qwen2.5-coder:32b", verbose: bool = False):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1
        )
        self.verbose = verbose
        
    async def parse(self, query: str) -> QueryIntent:
        """Parse a natural language query into structured intent with semantic expansion"""
        try:
            companies = CompanyRegistry.get_all_companies()
            prompt = QUERY_TEMPLATE.format(
                companies=", ".join(companies),
                query=query
            )
            
            response = await self.llm.ainvoke(prompt)
            
            if isinstance(response, str):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    response = response[start:end]
                    
            intent_data = json.loads(response)
            
            # Build enhanced search topic using semantic context
            semantic_context = intent_data.get('semantic_context', {})
            enhanced_topic = self._build_enhanced_topic(
                intent_data['topic'],
                semantic_context
            )
            intent_data['topic'] = enhanced_topic
            
            # Update keywords with related terms
            related_terms = (
                semantic_context.get('core_concepts', []) +
                semantic_context.get('related_terms', []) +
                semantic_context.get('aspects', [])
            )
            intent_data['filters']['keywords'].extend(related_terms)
            
            # Only print in verbose mode
            if self.verbose:
                print(f"Parsed intent: {json.dumps(intent_data, indent=2)}")
            
            return QueryIntent.model_validate(intent_data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse query intent: {str(e)}")
            
    def _build_enhanced_topic(self, base_topic: str, semantic_context: dict) -> str:
        """Build an enhanced topic description for better semantic search"""
        parts = [base_topic]
        
        if semantic_context.get('core_concepts'):
            concepts = ', '.join(semantic_context['core_concepts'])
            parts.append(f"Including core concepts: {concepts}")
            
        if semantic_context.get('aspects'):
            aspects = ', '.join(semantic_context['aspects'])
            parts.append(f"Considering aspects like: {aspects}")
            
        return ' '.join(parts)


class IntentParser:
    def __init__(self, model_name: str = "qwen2.5-coder:32b", verbose: bool = False):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1
        )
        self.verbose = verbose
        
    async def parse(self, query: str) -> QueryIntent:
        """Parse a natural language query into structured intent"""
        try:
            companies = CompanyRegistry.get_all_companies()
            prompt = QUERY_TEMPLATE.format(
                companies=", ".join(companies),
                query=query
            )
            
            response = await self.llm.ainvoke(prompt)
            
            if isinstance(response, str):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    response = response[start:end]
                    
            intent_data = json.loads(response)
            
            if 'filters' in intent_data and 'companies' in intent_data['filters']:
                intent_data['filters']['companies'] = [
                    company for company in intent_data['filters']['companies']
                    if company in CompanyRegistry.COMPANIES
                ]
            
            # Only print in verbose mode
            if self.verbose:
                print(f"Parsed intent: {json.dumps(intent_data, indent=2)}")
            
            return QueryIntent.model_validate(intent_data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse query intent: {str(e)}")