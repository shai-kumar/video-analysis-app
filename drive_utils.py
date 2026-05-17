import os
import pickle
import gdown
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import io

SCOPES = ['https://www.googleapis.com/auth/drive']

def get_drive_service(credentials_file='credentials.json', token_file='token.pickle'):
    creds = None
    if os.path.exists(token_file):
        with open(token_file, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                return None
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def list_drive_videos(service):
    try:
        # Search for video files that aren't in the trash
        results = service.files().list(
            q="mimeType contains 'video/' and trashed = false",
            pageSize=50, 
            fields="nextPageToken, files(id, name)",
            orderBy="modifiedTime desc"
        ).execute()
        return results.get('files', [])
    except Exception as e:
        print(f"An error occurred listing files: {e}")
        return []

def download_file_from_drive(service, file_id, output_path):
    try:
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(output_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
        return output_path
    except Exception as e:
        print(f"An error occurred downloading file: {e}")
        return None

def upload_file_to_drive(service, file_path, name):
    try:
        file_metadata = {'name': name}
        media = MediaFileUpload(file_path, mimetype='video/mp4')
        file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        return file.get('id')
    except Exception as e:
        print(f"An error occurred uploading file: {e}")
        return None

def download_public_link(url, output_path):
    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        return output_path
    except Exception as e:
        print(f"An error occurred downloading public link: {e}")
        return None
