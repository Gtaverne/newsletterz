from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import pickle

CREDENTIALS_PATH = '../../secrets/credentials.json'


SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']


def get_gmail_service():
    flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_PATH, SCOPES)
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)

def fetch_recent_emails(service, max_results=10):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    
    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        subject = next((header['value'] for header in msg['payload']['headers'] 
                       if header['name'].lower() == 'subject'), 'No Subject')
        print(f"Subject: {subject}")

if __name__ == '__main__':
    service = get_gmail_service()
    fetch_recent_emails(service)