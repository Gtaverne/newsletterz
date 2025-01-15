import pytest
from src.search.response_crafter import ResponseCrafter
from datetime import datetime

@pytest.mark.asyncio
async def test_count_response():
    crafter = ResponseCrafter()
    
    # Test count query
    count_results = {
        "type": "count",
        "count": 15,
        "total_results": 15,
        "results": [
            {
                "from": "newsletter@mckinsey.com",
                "company": "mckinsey",
                "date": int(datetime(2024, 1, 1).timestamp()),
                "subject": "AI Trends Report",
                "content": "Our analysis shows increasing adoption of AI...",
                "distance": 0.15
            }
        ]
    }
    
    response = await crafter.craft_response(
        "how many emails about AI from McKinsey?",
        count_results
    )
    print("\nCount Response Test:")
    print(response)
    assert response is not None
    assert "15" in response
    assert "McKinsey" in response

@pytest.mark.asyncio
async def test_trend_analysis():
    crafter = ResponseCrafter()
    
    # Test trend analysis
    trend_results = {
        "type": "trend",
        "total_results": 3,
        "results": [
            {
                "from": "news@bcg.com",
                "company": "bcg",
                "date": int(datetime(2024, 1, 15).timestamp()),
                "subject": "Digital Transformation in 2024",
                "content": "Companies are increasingly focusing on AI adoption...",
                "distance": 0.2
            },
            {
                "from": "insights@mckinsey.com",
                "company": "mckinsey",
                "date": int(datetime(2024, 1, 10).timestamp()),
                "subject": "AI Implementation Challenges",
                "content": "Organizations face several hurdles in AI adoption...",
                "distance": 0.25
            },
            {
                "from": "research@bain.com",
                "company": "bain",
                "date": int(datetime(2024, 1, 5).timestamp()),
                "subject": "Tech Trends 2024",
                "content": "AI remains a top priority for executives...",
                "distance": 0.3
            }
        ]
    }
    
    response = await crafter.craft_response(
        "what are the trends in AI adoption from consulting firms?",
        trend_results
    )
    print("\nTrend Analysis Test:")
    print(response)
    assert response is not None
    assert any(company in response.lower() for company in ["bcg", "mckinsey", "bain"])
    assert "AI" in response

@pytest.mark.asyncio
async def test_error_handling():
    crafter = ResponseCrafter()
    
    # Test error case
    error_results = {
        "type": "error",
        "message": "Search execution failed"
    }
    
    response = await crafter.craft_response(
        "invalid query",
        error_results
    )
    print("\nError Handling Test:")
    print(response)
    assert "error" in response.lower()
    assert "failed" in response.lower()

@pytest.mark.asyncio
async def test_empty_results():
    crafter = ResponseCrafter()
    
    # Test empty results
    empty_results = {
        "type": "empty",
        "total_results": 0,
        "results": []
    }
    
    response = await crafter.craft_response(
        "emails about quantum computing from startups",
        empty_results
    )
    print("\nEmpty Results Test:")
    print(response)
    assert "no emails found" in response.lower()

async def run_all_tests():
    """Run all tests sequentially"""
    print("Starting Response Crafter Tests...")
    
    print("\n=== Testing Count Response ===")
    await test_count_response()
    
    print("\n=== Testing Trend Analysis ===")
    await test_trend_analysis()
    
    print("\n=== Testing Error Handling ===")
    await test_error_handling()
    
    print("\n=== Testing Empty Results ===")
    await test_empty_results()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_all_tests())