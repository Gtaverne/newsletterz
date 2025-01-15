import asyncio
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
import argparse
import sys
import signal
from datetime import datetime
from src.search.intent_parser import IntentParser
from src.search.search_executor import SearchExecutor
from src.search.response_crafter import ResponseCrafter
from src.utils.logging import logger

from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

class SearchInterface:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.display_limit = 20
        self.search_limit = 500
        self.is_running = True
        self.console = Console()

    async def execute_search(self, query: str) -> None:
        """Execute email search with natural language query"""
        try:
            # Initialize components with verbose flag
            intent_parser = IntentParser(verbose=self.verbose)
            search_executor = SearchExecutor(verbose=self.verbose)
            response_crafter = ResponseCrafter(verbose=self.verbose)
            
            # Step 1: Parse Intent with spinner
            with self.console.status("[bold green]Reformulating your question...", spinner="dots"):
                intent = await intent_parser.parse(query)
            
            # Step 2: Execute Search with spinner
            with self.console.status("[bold green]Fetching relevant emails...", spinner="dots"):
                results = await search_executor.execute_search(intent, limit=self.search_limit)
                self.console.print(f"Found {results.get('total_results')} matching emails")
            
            # Format display results
            if results.get('results'):
                display_results = results.copy()
                display_results['results'] = results['results'][:self.display_limit]
            else:
                display_results = results
            
            # Show results based on type
            if results['type'] == 'error':
                self.console.print(f"[red]Error:[/red] {results['message']}")
                return
                
            # Step 3: Craft Response with spinner
            with self.console.status("[bold green]Summarizing...", spinner="dots"):
                response = await response_crafter.craft_response(query, results)
            
            # Print final results
            self.console.print("\n[bold green]Summary:[/bold green]")
            self.console.print(response)
            
            if display_results.get('results'):
                self.console.print("\n[bold green]Matching Emails:[/bold green]")
                self.console.print(self.format_results_table(display_results))
                if self.verbose and len(results.get('results', [])) > self.display_limit:
                    self.console.print(f"Showing {self.display_limit} of {len(results['results'])} results")
                
        except Exception as e:
            self.console.print(f"[red]Error:[/red] {str(e)}")

    def format_results_table(self, results: dict) -> Table:
        """Format search results into a rich table"""
        if results.get('type') == 'error':
            return Panel(f"[red]Error:[/red] {results['message']}", title="Search Error")
            
        if results.get('type') == 'count':
            return Panel(f"Found {results['total_results']} matching emails", title="Search Results")
        
        if not results.get('results'):
            return Panel("[yellow]No results found[/yellow]", title="Search Results")
        
        table = Table(title="Search Results", show_header=True, header_style="bold magenta")
        table.add_column("Date", style="cyan", no_wrap=True)
        table.add_column("From", style="green")
        table.add_column("Subject", style="blue")
        
        for result in results['results']:
            date = datetime.fromtimestamp(result['date']).strftime('%Y-%m-%d')
            sender = f"{result['from']} ({result['company']})"
            table.add_row(date, sender, result['subject'])
        
        return table

    async def run(self) -> None:
        """Run the interactive interface"""
        self.console.print("[bold blue]Email Search Interface[/bold blue]")
        self.console.print("Type your query or /? for help\n")
        
        while self.is_running:
            try:
                query = Prompt.ask(">>> ")
                query = query.strip()
                
                if not query:
                    continue
                    
                if query.lower() in ['/quit', '/q']:
                    self.console.print("[green]Goodbye![/green]")
                    break
                elif query.lower() in ['/help', '/?']:
                    self.print_help()
                elif query.lower().startswith('/limit '):
                    try:
                        new_limit = int(query.split()[1])
                        self.display_limit = min(new_limit, 50)
                        self.console.print(f"[green]Display limit set to {self.display_limit}[/green]")
                    except ValueError:
                        self.console.print("[red]Invalid limit value[/red]")
                else:
                    await self.execute_search(query)
                    
            except KeyboardInterrupt:
                self.console.print("\n[green]Goodbye![/green]")
                break
            except EOFError:  # Handle Ctrl+D
                self.console.print("\n[green]Goodbye![/green]")
                break
            except Exception as e:
                self.console.print(f"[red]Error:[/red] {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Interactive email search interface')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    def handle_interrupt(signum, frame):
        print("\n[green]Goodbye![/green]")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, handle_interrupt)
    
    interface = SearchInterface(verbose=args.verbose)
    try:
        asyncio.run(interface.run())
    except KeyboardInterrupt:
        print("\n[green]Goodbye![/green]")
        sys.exit(0)

if __name__ == '__main__':
    main()