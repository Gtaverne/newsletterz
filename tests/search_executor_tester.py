import asyncio
import pytest
from datetime import datetime, timedelta
from src.search.models import QueryIntent, FilterConfig, TimeRange
from src.search.search_executor import SearchExecutor
from src.search.company_registry import CompanyRegistry

@pytest.mark.asyncio
async def debug_chrome_content():
    """Print what's actually in the database"""
    executor = SearchExecutor()
    
    # Get all documents
    results = executor.collection.get(
        include=['metadatas'],
        limit=5
    )
    
    print("\nChromaDB Content Sample:")
    for metadata in results['metadatas']:
        print(f"From: {metadata.get('from')}")
        print(f"Company: {metadata.get('company')}")
        print(f"Subject: {metadata.get('subject')}")
        print("---")

@pytest.mark.asyncio
async def test_basic_search():
    executor = SearchExecutor()
    
    # Simple search for AI content
    intent = QueryIntent(
        type="list",
        topic="artificial intelligence",
        filters=FilterConfig(
            keywords=["AI", "machine learning"]
        ),
        reasoning="Test basic search"
    )
    
    result = await executor.execute_search(intent, limit=5)
    print("\nBasic Search Results:")
    print(f"Type: {result['type']}")
    print(f"Total results: {result.get('total_results', 0)}")
    if result.get('results'):
        for r in result['results'][:2]:
            print(f"\nSubject: {r['subject']}")
            print(f"From: {r['from']}")
            print(f"Company: {r.get('company', 'unknown')}")
            print(f"Relevance: {r['relevance']:.3f}")
            print(f"Preview: {r['content'][:100]}...")

@pytest.mark.asyncio
async def test_company_groups():
    executor = SearchExecutor()
    
    # Test company groups
    intents = [
        # Test MBB
        QueryIntent(
            type="list",
            topic="digital transformation",
            filters=FilterConfig(
                companies=["mckinsey", "bcg", "bain"],  # MBB
                keywords=["digital"]
            ),
            reasoning="Test MBB search"
        ),
        # Test Big 4
        QueryIntent(
            type="list",
            topic="digital transformation",
            filters=FilterConfig(
                companies=["deloitte", "pwc", "ey", "kpmg"],  # Big 4
                keywords=["digital"]
            ),
            reasoning="Test Big 4 search"
        )
    ]
    
    for intent in intents:
        result = await executor.execute_search(intent, limit=5)
        print(f"\nCompany Group Search Results ({','.join(intent.filters.companies)}):")
        print(f"Total results: {result.get('total_results', 0)}")
        if result.get('results'):
            for r in result['results'][:2]:
                print(f"\nFrom: {r['from']}")
                print(f"Company: {r.get('company', 'unknown')}")
                print(f"Subject: {r['subject']}")
                print(f"Relevance: {r['relevance']:.3f}")

@pytest.mark.asyncio
async def test_date_filtered_search():
    executor = SearchExecutor()
    
    # Search with date range
    start_date = datetime.now() - timedelta(days=90)
    end_date = datetime.now()
    
    intent = QueryIntent(
        type="list",
        topic="cloud computing",
        filters=FilterConfig(
            companies=["deloitte"],  # Single company
            time_range=TimeRange(
                start=start_date,
                end=end_date,
                description=f"Last 90 days"
            )
        ),
        reasoning="Test date filtered search"
    )
    
    result = await executor.execute_search(intent, limit=5)
    print("\nDate Filtered Search Results:")
    print(f"Time range: Last 90 days")
    print(f"Total results: {result.get('total_results', 0)}")
    if result.get('results'):
        for r in result['results'][:2]:
            print(f"\nDate: {r['date']}")
            print(f"Subject: {r['subject']}")
            print(f"From: {r['from']}")
            print(f"Company: {r.get('company', 'unknown')}")

@pytest.mark.asyncio
async def test_international_orgs():
    executor = SearchExecutor()
    
    # Test international organizations
    intent = QueryIntent(
        type="list",
        topic="climate change",
        filters=FilterConfig(
            companies=["imf", "un", "idb"],
            keywords=["climate"]
        ),
        reasoning="Test international orgs search"
    )
    
    result = await executor.execute_search(intent, limit=5)
    print("\nInternational Orgs Search Results:")
    print(f"Total results: {result.get('total_results', 0)}")
    if result.get('results'):
        for r in result['results'][:2]:
            print(f"\nFrom: {r['from']}")
            print(f"Company: {r.get('company', 'unknown')}")
            print(f"Subject: {r['subject']}")
            print(f"Relevance: {r['relevance']:.3f}")

@pytest.mark.asyncio
async def test_tech_companies():
    executor = SearchExecutor()
    
    # Test FAANG companies
    intent = QueryIntent(
        type="list",
        topic="artificial intelligence",
        filters=FilterConfig(
            companies=["meta", "google", "amazon"],
            keywords=["AI"]
        ),
        reasoning="Test tech companies search"
    )
    
    result = await executor.execute_search(intent, limit=5)
    print("\nTech Companies Search Results:")
    print(f"Total results: {result.get('total_results', 0)}")
    if result.get('results'):
        for r in result['results'][:2]:
            print(f"\nFrom: {r['from']}")
            print(f"Company: {r.get('company', 'unknown')}")
            print(f"Subject: {r['subject']}")
            print(f"Relevance: {r['relevance']:.3f}")

@pytest.mark.asyncio
async def test_negative_search():
    executor = SearchExecutor()
    # Search results from mck's competitors

async def run_all_tests():
    """Run all tests sequentially"""
    print("Starting Search Executor Tests...")
    print(f"Available companies: {', '.join(CompanyRegistry.get_all_companies())}")

    print("\nDebugging Chrome Content...")
    await debug_chrome_content()
    
    print("\n=== Testing Basic Search ===")
    await test_basic_search()
    
    print("\n=== Testing Company Groups ===")
    await test_company_groups()
    
    print("\n=== Testing International Organizations ===")
    await test_international_orgs()
    
    print("\n=== Testing Tech Companies ===")
    await test_tech_companies()
    
    print("\n=== Testing Date Filtered Search ===")
    await test_date_filtered_search()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(run_all_tests())