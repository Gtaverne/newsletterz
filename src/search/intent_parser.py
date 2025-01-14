from langchain_ollama import OllamaLLM
from typing import Optional
import json
from datetime import datetime
from .models import QueryIntent, FilterConfig, TimeRange
from .company_registry import CompanyRegistry

QUERY_TEMPLATE = """You are an AI assistant specialized in analyzing email search queries.
Available companies: {companies}

Common company groupings:
- MBB (Top strategy firms): mckinsey, bcg, bain
- Big 4: deloitte, pwc, ey, kpmg
- FAANG: meta, apple, amazon, netflix, google
- International organizations: imf, idb, un

Analyze this search query: "{query}"

Think through step by step:
1. What is the primary intent?
   - Is it asking for a count? ("how many...")
   - Is it asking for a summary/trends? ("what are the trends...")
   - Is it asking for a list? ("show me...")

2. What is the main topic?
   - Extract the core subject matter
   - Identify related keywords

3. Identify companies:
   - Look for explicit company names
   - Understand business terms and translations ("MBB" → mckinsey, bcg, bain, "ONU" → un, "FMI" → imf)
   - Map industry groups ("FAANG", "Big 4") to individual companies
   - Only use companies from our available list

Output a JSON object that matches this schema:
{{
    "type": "count" | "summary" | "trend" | "list",
    "topic": "main topic or subject",
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

Examples of company mapping:
- "MBB" → ["mckinsey", "bcg", "bain"]
- "FAANG views on AI" → ["meta", "apple", "amazon", "netflix", "google"]
- "big 4 consulting" → ["deloitte", "pwc", "ey", "kpmg"]
- "What does McKinsey say" → ["mckinsey"]"""


class IntentParser:
    def __init__(self, model_name: str = "qwen2.5-coder:32b"):
        """Initialize with specified model"""
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.1
        )
        
    async def parse(self, query: str) -> QueryIntent:
        """Parse a natural language query into structured intent"""
        try:
            # Get list of known companies
            companies = CompanyRegistry.get_all_companies()
            
            # Format the prompt with companies
            prompt = QUERY_TEMPLATE.format(
                companies=", ".join(companies),
                query=query
            )
            
            # Get response from LLM
            response = await self.llm.ainvoke(prompt)
            
            # Extract JSON from response if needed
            if isinstance(response, str):
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    response = response[start:end]
                    
            # Parse into QueryIntent
            intent_data = json.loads(response)
            
            # Validate companies - ensure we only get keys from our registry
            if 'filters' in intent_data and 'companies' in intent_data['filters']:
                intent_data['filters']['companies'] = [
                    company for company in intent_data['filters']['companies']
                    if company in CompanyRegistry.COMPANIES  # Direct key match, no lower()
                ]
            
            # Debug print
            print(f"Parsed intent: {json.dumps(intent_data, indent=2)}")
            
            return QueryIntent.model_validate(intent_data)
            
        except Exception as e:
            raise ValueError(f"Failed to parse query intent: {str(e)}")