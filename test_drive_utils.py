import unittest
from unittest.mock import patch, MagicMock
import sys

# Mock missing modules
sys.modules['gdown'] = MagicMock()
sys.modules['googleapiclient'] = MagicMock()
sys.modules['googleapiclient.discovery'] = MagicMock()
sys.modules['googleapiclient.http'] = MagicMock()
sys.modules['google_auth_oauthlib'] = MagicMock()
sys.modules['google_auth_oauthlib.flow'] = MagicMock()
sys.modules['google.auth'] = MagicMock()
sys.modules['google.auth.transport'] = MagicMock()
sys.modules['google.auth.transport.requests'] = MagicMock()

import drive_utils

class TestDriveUtils(unittest.TestCase):

    @patch('drive_utils.build')
    @patch('drive_utils.os.path.exists')
    @patch('drive_utils.pickle.load')
    @patch('builtins.open')
    def test_get_drive_service_with_valid_token(self, mock_open, mock_pickle_load, mock_exists, mock_build):
        # Setup mock for token exists and valid
        mock_exists.side_effect = lambda x: x == 'token.pickle'
        mock_creds = MagicMock()
        mock_creds.valid = True
        mock_pickle_load.return_value = mock_creds
        
        mock_service = MagicMock()
        mock_build.return_value = mock_service

        service = drive_utils.get_drive_service(credentials_file='dummy.json', token_file='token.pickle')
        
        self.assertEqual(service, mock_service)
        mock_build.assert_called_once_with('drive', 'v3', credentials=mock_creds)

    def test_list_drive_videos(self):
        mock_service = MagicMock()
        mock_files = mock_service.files.return_value
        mock_list = mock_files.list.return_value
        mock_list.execute.return_value = {
            'files': [
                {'id': '1', 'name': 'video1.mp4'},
                {'id': '2', 'name': 'video2.webm'}
            ]
        }

        videos = drive_utils.list_drive_videos(mock_service)
        
        self.assertEqual(len(videos), 2)
        self.assertEqual(videos[0]['name'], 'video1.mp4')
        mock_service.files().list.assert_called_once()

    @patch('drive_utils.io.FileIO')
    @patch('drive_utils.MediaIoBaseDownload')
    def test_download_file_from_drive(self, mock_downloader_class, mock_file_io):
        mock_service = MagicMock()
        mock_request = MagicMock()
        mock_service.files().get_media.return_value = mock_request

        mock_downloader = MagicMock()
        mock_downloader.next_chunk.side_effect = [(None, False), (None, True)]
        mock_downloader_class.return_value = mock_downloader

        output_path = drive_utils.download_file_from_drive(mock_service, 'file_id_123', 'out.mp4')

        self.assertEqual(output_path, 'out.mp4')
        mock_service.files().get_media.assert_called_once_with(fileId='file_id_123')
        mock_downloader_class.assert_called_once()

    @patch('drive_utils.gdown.download')
    def test_download_public_link(self, mock_gdown_download):
        result = drive_utils.download_public_link('http://example.com/file', 'out.mp4')
        
        self.assertEqual(result, 'out.mp4')
        mock_gdown_download.assert_called_once_with('http://example.com/file', 'out.mp4', quiet=False, fuzzy=True)

if __name__ == '__main__':
    unittest.main()
