#!/usr/bin/env python3
import asyncio
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from datetime import datetime
from src.llm.enhanced_search_chain import EnhancedEmailSearchChain

def create_results_table(results, query_type):
    """Create a rich table or panel based on query type"""
    if query_type == 'count':
        return Panel(f"Found {results['count']} matching emails", title="Search Results")
    
    table = Table()
    table.add_column("Date", style="cyan")
    table.add_column("From", style="green")
    table.add_column("Subject", style="blue")
    table.add_column("Preview")
    
    for result in results['results']:
        table.add_row(
            result['date'],
            result['from'],
            result['subject'],
            result['preview']
        )
    
    return table

async def main():
    parser = argparse.ArgumentParser(description='Search emails using natural language')
    parser.add_argument('query', help='Natural language search query')
    parser.add_argument('--limit', type=int, default=10000, help='Maximum number of results')
    args = parser.parse_args()
    
    console = Console()
    
    try:
        # Initialize search chain
        search_chain = EnhancedEmailSearchChain()
        
        # Execute search
        with console.status("[bold green]Searching emails..."):
            results = await search_chain.search(args.query, args.limit)
        
        # Handle errors
        if results.get('type') == 'error':
            console.print(f"[red]Error:[/red] {results['message']}")
            return
        
        if not results.get('results', []) and not results.get('count'):
            console.print("[yellow]No results found[/yellow]")
            return
        
        # Display results
        console.print(create_results_table(results, results['type']))
        
        if results.get('results'):
            console.print(f"\nFound {len(results['results'])} results")
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())