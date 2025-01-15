import unittest
from datetime import datetime
import os
from src.email.gmail_fetcher import GmailFetcher
import json

class TestGmailFetcher(unittest.TestCase):
    def setUp(self):
        credentials_path = os.path.join(os.path.dirname(__file__), '../../secrets/credentials.json')
        self.fetcher = GmailFetcher(credentials_path)

    def test_fetch_emails_with_pagination(self):
        result = self.fetcher.fetch_emails(page_size=5)
        self.assertIsInstance(result['emails'], list)
        self.assertLessEqual(len(result['emails']), 5)
        
        # Test structure of fetched emails
        if result['emails']:
            email = result['emails'][0]
            self.assertIn('id', email)
            self.assertIn('thread_id', email)
            self.assertIn('internal_date', email)
            self.assertIn('headers', email)
            self.assertIn('clean_text', email)
            self.assertIn('raw_html', email)
            self.assertIn('urls', email)

    def test_fetch_emails_with_date_range(self):
        start = datetime(2024, 1, 1)
        result = self.fetcher.fetch_emails(start_date=start, page_size=5)
        self.assertIsInstance(result['emails'], list)

    def test_email_content_structure(self):
        """Test that emails are properly parsed and contain expected fields"""
        result = self.fetcher.fetch_emails(page_size=1)
        
        if result['emails']:
            email = result['emails'][0]
            
            # Print email structure for inspection
            print("\nEmail Structure:")
            print(json.dumps({
                'id': email['id'],
                'thread_id': email['thread_id'],
                'internal_date': email['internal_date'],
                'headers': {
                    'subject': email['headers'].get('subject'),
                    'from': email['headers'].get('from'),
                    'date': email['headers'].get('date')
                },
                'text_preview': email['clean_text'][:200] + "...",  # First 200 chars
                'urls_count': len(email['urls'])
            }, indent=2))
            
            # Test required fields
            self.assertTrue(email['clean_text'])  # Should not be empty
            self.assertTrue(email['raw_html'])    # Should not be empty
            self.assertIsInstance(email['urls'], list)
            
            # Test headers
            required_headers = ['subject', 'from', 'date']
            for header in required_headers:
                self.assertIn(header, email['headers'].keys())

if __name__ == '__main__':
    unittest.main()