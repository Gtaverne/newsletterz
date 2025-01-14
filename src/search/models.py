from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime

class TimeRange(BaseModel):
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    description: str = Field(description="Human readable description of the time range")

class FilterConfig(BaseModel):
    companies: List[str] = Field(default_factory=list, description="Company domains or business terms")
    time_range: Optional[TimeRange] = None
    keywords: List[str] = Field(default_factory=list, description="Additional keywords for filtering")
    
class QueryIntent(BaseModel):
    type: Literal["count", "summary", "trend", "list"] = Field(
        description="Type of query requested"
    )
    topic: str = Field(description="Main topic or subject of the query")
    filters: FilterConfig = Field(default_factory=FilterConfig)
    reasoning: str = Field(description="Explanation of why this intent was chosen")

class EmailReference(BaseModel):
    id: str
    subject: str
    sender: str
    date: datetime
    relevance_score: float
    preview: Optional[str] = None

class SearchResponse(BaseModel):
    text: str = Field(description="Generated response text")
    confidence: float = Field(ge=0, le=1, description="Confidence score of the response")
    sources: List[EmailReference] = Field(default_factory=list, description="Supporting email references")
    metadata: dict = Field(default_factory=dict, description="Additional metadata about the search")