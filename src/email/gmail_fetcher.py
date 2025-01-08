from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os
import base64
from bs4 import BeautifulSoup
from datetime import datetime

class GmailFetcher:
    SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
    
    def __init__(self, credentials_path):
        self.credentials_path = credentials_path
        self.service = self._get_gmail_service()
    
    def _get_gmail_service(self):
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, self.SCOPES)
        creds = flow.run_local_server(port=0)
        return build('gmail', 'v1', credentials=creds)

    def _extract_clean_text(self, content, content_type='text/plain'):
        """Extract readable text from content based on type"""
        if not content:
            return ""
            
        if content_type == 'text/html':
            soup = BeautifulSoup(content, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            # Get text while preserving some structure
            lines = (line.strip() for line in soup.get_text(separator='\n').splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            return '\n'.join(chunk for chunk in chunks if chunk)
        else:
            # For plain text, just clean up whitespace
            lines = content.splitlines()
            return '\n'.join(line.strip() for line in lines if line.strip())

    def _extract_urls(self, html_content):
        """Extract URLs from HTML content"""
        if not html_content:
            return []
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            return [a.get('href') for a in soup.find_all('a', href=True)]
        except:
            return []

    def _get_part_content(self, part):
        """Extract content from a message part"""
        if 'body' not in part:
            return None
        if 'data' in part['body']:
            return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        return None

    def _parse_message_parts(self, payload):
        """Recursively parse message parts"""
        html_content = None
        text_content = None
        
        if 'parts' in payload:
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                if mime_type == 'text/html':
                    html_content = self._get_part_content(part)
                elif mime_type == 'text/plain':
                    text_content = self._get_part_content(part)
                # Handle nested multipart
                elif 'parts' in part:
                    nested_html, nested_text = self._parse_message_parts(part)
                    html_content = html_content or nested_html
                    text_content = text_content or nested_text
        else:
            # Single part message
            mime_type = payload.get('mimeType', '')
            content = self._get_part_content(payload)
            if mime_type == 'text/html':
                html_content = content
            elif mime_type == 'text/plain':
                text_content = content

        return html_content, text_content

    def _parse_email(self, msg):
        """Parse a single email into a clean structure"""
        try:
            # Extract headers
            headers = {}
            for header in msg['payload']['headers']:
                headers[header['name'].lower()] = header['value']
            
            # Get content from parts
            html_content, text_content = self._parse_message_parts(msg['payload'])
            
            # Prefer HTML content for better structure
            content_for_text = html_content if html_content else text_content
            content_type = 'text/html' if html_content else 'text/plain'
            
            if not content_for_text:
                content_for_text = "Empty email body"
            
            clean_text = self._extract_clean_text(content_for_text, content_type)
            urls = self._extract_urls(html_content) if html_content else []
            
            return {
                "id": msg['id'],
                "thread_id": msg['threadId'],
                "internal_date": datetime.fromtimestamp(int(msg['internalDate'])/1000).isoformat(),
                "headers": headers,
                "clean_text": clean_text,
                "urls": urls
            }
        except Exception as e:
            raise Exception(f"Error parsing email {msg.get('id', 'unknown')}: {str(e)}")

    def fetch_emails(self, start_date=None, end_date=None, page_size=10, page_token=None):
        """Fetch emails with pagination support"""
        query = self._build_date_query(start_date, end_date)
        
        results = self.service.users().messages().list(
            userId='me',
            maxResults=page_size,
            pageToken=page_token,
            q=query
        ).execute()

        emails = []
        for message in results.get('messages', []):
            try:
                msg = self.service.users().messages().get(userId='me', id=message['id']).execute()
                emails.append(self._parse_email(msg))
            except Exception as e:
                print(f"Failed to process message {message['id']}: {e}")
                continue

        return {
            'emails': emails,
            'next_page_token': results.get('nextPageToken')
        }

    def _build_date_query(self, start_date, end_date):
        query_parts = []
        if start_date:
            query_parts.append(f'after:{start_date.strftime("%Y/%m/%d")}')
        if end_date:
            query_parts.append(f'before:{end_date.strftime("%Y/%m/%d")}')
        return ' '.join(query_parts)