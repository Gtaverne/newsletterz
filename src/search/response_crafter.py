from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, List
from datetime import datetime

class ResponseCrafter:
    def __init__(self, model_name: str = "qwen2.5-coder:32b", limit: int = 100, verbose: bool = False):
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.3
        )
        self.verbose = verbose
        self.limit = limit

    def _get_prompt(self, query: str, results: Dict) -> str:
        query_type = results['type']
        total_results = results['total_results']

        if query_type == "count":
            # For count, we just need one example email
            example = results['results'][0] if results['results'] else None
            if example:
                date = datetime.fromtimestamp(example['date']).strftime('%Y-%m-%d')
                return f"""The user asked: "{query}"
There are {total_results} matching emails.
Example: On {date}, subject "{example['subject']}"
Format the response as a natural sentence stating the count and the example."""
            return f"There are {total_results} matching emails."

        # For other types, select content based on intent
        filtered_results = []
        if query_type == "trend":
            # Group by company and take most recent from each
            by_company = {}
            for r in results['results']:
                company = r['company']
                if company not in by_company or r['date'] > by_company[company]['date']:
                    by_company[company] = r
            filtered_results = list(by_company.values())

        elif query_type == "summary":
            # Take top results by relevance (lowest distance)
            filtered_results = sorted(results['results'], key=lambda x: x['distance'])[:3]

        else:  # list
            # Take top results
            filtered_results = results['results'][:self.limit]

        # Format the selected results
        details = []
        for r in filtered_results:
            date = datetime.fromtimestamp(r['date']).strftime('%Y-%m-%d')
            details.append(f"""From: {r['from']} ({r['company']})
Date: {date}
Subject: {r['subject']}
Content: {r['content']}""")

        return f"""The user asked: "{query}"
Based on {total_results} matching emails, here are the most relevant:

{chr(10).join(details)}

Provide a concise {query_type} focusing on the key points. Use quotes when relevant."""

    async def craft_response(self, query: str, search_results: Dict) -> str:
        if search_results.get('type') == 'error':
            return f"Error: {search_results.get('message')}"
            
        if search_results.get('type') == 'empty' or search_results.get('total_results', 0) == 0:
            return "No emails found matching your query."

        if self.verbose:
            print(f"\nCrafting response for {search_results['type']} query")

        prompt = self._get_prompt(query, search_results)
        response = await self.llm.ainvoke(prompt)
        
        return response.strip()