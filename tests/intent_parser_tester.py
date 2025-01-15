import pytest
import asyncio
from datetime import datetime
from src.search.intent_parser import IntentParser
from src.search.models import QueryIntent

pytestmark = pytest.mark.asyncio

async def test_count_query():
    parser = IntentParser()
    query = "how many emails about AI from McKinsey?"
    
    result = await parser.parse(query)
    print(f"\nResult for count query: {result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "count"
    assert any(keyword.lower() in ["ai", "artificial intelligence"] 
              for keyword in result.filters.keywords)
    assert any("mckinsey" in company.lower() 
              for company in result.filters.companies)
    
async def test_trend_query():
    parser = IntentParser()
    query = "what are the latest trends in cloud computing from big consulting firms?"
    
    result = await parser.parse(query)
    print(f"\nResult for trend query: {result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "trend"
    assert "cloud" in result.topic.lower()
    
async def test_summary_with_date():
    parser = IntentParser()
    query = "summarize discussions about machine learning from September 2024"
    
    result = await parser.parse(query)
    print(f"\nResult for summary query: {result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "summary"
    assert "machine learning" in result.topic.lower()

async def test_multiple_companies():
    parser = IntentParser()
    query = "show me AI trends from McKinsey and BCG in 2024"
    
    result = await parser.parse(query)
    print(f"\nResult for multiple companies:\n{result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "trend"
    assert len(result.filters.companies) >= 2
    assert any("mckinsey" in company.lower() for company in result.filters.companies)
    assert any("bcg" in company.lower() for company in result.filters.companies)

async def test_business_terms():
    parser = IntentParser()
    query = "summarize what big consulting firms say about digital transformation"
    
    result = await parser.parse(query)
    print(f"\nResult for business terms:\n{result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "summary"
    assert "digital transformation" in result.topic.lower()
    # The model should recognize 'big consulting firms' as a business term
    assert len(result.filters.companies) > 0
    
async def test_complex_time_range():
    parser = IntentParser()
    query = "how many cloud computing emails from Deloitte between March and September 2024?"
    
    result = await parser.parse(query)
    print(f"\nResult for time range:\n{result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert result.type == "count"
    assert result.filters.time_range is not None
    assert "deloitte" in [company.lower() for company in result.filters.companies]

async def test_multiple_topics():
    parser = IntentParser()
    query = "show me emails about AI and machine learning from big tech companies"
    
    result = await parser.parse(query)
    print(f"\nResult for multiple topics:\n{result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert len(result.filters.keywords) >= 2
    assert any(keyword.lower() in ["ai", "artificial intelligence"] 
              for keyword in result.filters.keywords)
    assert any("machine learning" in keyword.lower() 
              for keyword in result.filters.keywords)
    
async def test_negative_search():
    parser = IntentParser()
    query = "what are McKinsey's competitors saying about AI?"
    
    result = await parser.parse(query)
    print(f"\nResult for negative search:\n{result.model_dump_json(indent=2)}")
    
    assert isinstance(result, QueryIntent)
    assert "mckinsey" not in result.filters.companies
    assert any(company in result.filters.companies for company in ["bcg", "bain", "deloitte", "pwc", "ey", "kpmg"])
    assert "ai" in result.topic.lower()

if __name__ == "__main__":
    # Run a single test with more detailed output
    async def run_test():
        print("Running count query test...")
        await test_count_query()

        print("\n=== Testing Multiple Companies ===")
        await test_multiple_companies()
        
        print("\n=== Testing Business Terms ===")
        await test_business_terms()
        
        print("\n=== Testing Complex Time Range ===")
        await test_complex_time_range()
        
        print("\n=== Testing Multiple Topics ===")
        await test_multiple_topics()

        print("\n=== Testing Negative Search ===")
        await test_negative_search()
        
        print("\nAll tests completed successfully!")
        
        
    asyncio.run(run_test())