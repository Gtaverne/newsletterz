class CompanyRegistry:
    """Registry mapping company keys to their domain patterns and variations"""
    
    COMPANIES = {
        # Strategy consulting
        "mckinsey": {
            "domains": ["mckinsey.com", "email.mckinsey.com"],
            "patterns": ["mckinsey", "mck@"]
        },
        "bcg": {
            "domains": ["bcg.com", "email.bcg.com"],
            "patterns": ["@bcg", "boston consult"]
        },
        "bain": {
            "domains": ["bain.com"],
            "patterns": ["@bain", "bain &"]
        },
        
        # Big 4
        "deloitte": {
            "domains": ["deloitte.com", "email.deloitte.com"],
            "patterns": ["@deloitte"]
        },
        "pwc": {
            "domains": ["pwc.com"],
            "patterns": ["@pwc", "pricewaterhouse"]
        },
        "ey": {
            "domains": ["ey.com"],
            "patterns": ["@ey", "ernst & young"]
        },
        "kpmg": {
            "domains": ["kpmg.com"],
            "patterns": ["@kpmg"]
        },
        
        # Tech giants
        "meta": {
            "domains": ["meta.com", "fb.com", "facebook.com", "instagram.com", "whatsapp.com"],
            "patterns": ["@meta", "@fb", "@facebook", "@instagram", "@whatsapp"]
        },
        "apple": {
            "domains": ["apple.com"],
            "patterns": ["@apple"]
        },
        "amazon": {
            "domains": ["amazon.com", "aws.amazon.com", "aws.com"],
            "patterns": ["@amazon", "@aws"]
        },
        "netflix": {
            "domains": ["netflix.com"],
            "patterns": ["@netflix"]
        },
        "google": {
            "domains": ["google.com", "alphabet.com", "gmail.com"],
            "patterns": ["@google", "@alphabet"]
        },
        "microsoft": {
            "domains": ["microsoft.com", "ms.com"],
            "patterns": ["@microsoft", "@ms"]
        },
        
        # International organizations
        "imf": {
            "domains": ["imf.org", "internationalmonetaryfund.org"],
            "patterns": ["@imf", "international monetary fund"]
        },
        "idb": {
            "domains": ["iadb.org"],
            "patterns": ["@idb", "@iadb", "inter-american development bank"]
        },
        "un": {
            "domains": ["un.org", "undp.org", "unesco.org", "who.int"],
            "patterns": ["@un.org", "@undp", "@unesco", "@who.int", "united nations"]
        },
    }
    
    @classmethod
    def match_sender(cls, from_field: str) -> str:
        """
        Match sender against company patterns and domains
        
        Args:
            from_field: Email 'from' field value
            
        Returns:
            Matched company key or 'unknown'
        """
        if not from_field:
            return "unknown"
            
        from_field = from_field.lower()
        
        for company, matchers in cls.COMPANIES.items():
            # Check domains
            if any(domain in from_field for domain in matchers["domains"]):
                return company
                
            # Check patterns (more specific matches)
            if any(pattern in from_field for pattern in matchers["patterns"]):
                return company
        
        return "unknown"
    
    @classmethod
    def get_all_companies(cls) -> list[str]:
        """Return list of company keys for intent parsing"""
        return list(cls.COMPANIES.keys())

if __name__ == "__main__":
    # Test cases
    test_emails = [
        "someone@mckinsey.com",
        "person@email.mckinsey.com",
        "contact@internationalmonetaryfund.org",
        "news@imf.org",
        "updates@un.org",
        "noreply@undp.org",
        "info@deloitte.com",
        "fund@something.com",  # Should NOT match IMF
        "refund@company.com",  # Should NOT match IMF
        "someone@google.com",
        "noreply@aws.amazon.com"
    ]
    
    print("\nTesting company matching:")
    for email in test_emails:
        company = CompanyRegistry.match_sender(email)
        print(f"{email:40} -> {company}")