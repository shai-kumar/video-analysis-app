import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import os
import sys

# Mock requests module
sys.modules['requests'] = MagicMock()

import video_analyzer

class TestVideoAnalyzer(unittest.TestCase):

    @patch('shutil.which')
    @patch('os.path.exists')
    def test_get_executable_path_found(self, mock_exists, mock_which):
        # Case 1: Found by shutil.which
        mock_which.return_value = '/usr/bin/ffmpeg'
        path = video_analyzer.get_executable_path('ffmpeg')
        self.assertEqual(path, '/usr/bin/ffmpeg')
        
        # Case 2: Found by fallback paths
        mock_which.return_value = None
        mock_exists.side_effect = lambda x: x == '/opt/homebrew/bin/ffmpeg'
        path = video_analyzer.get_executable_path('ffmpeg')
        self.assertEqual(path, '/opt/homebrew/bin/ffmpeg')

        # Case 3: Not found
        mock_which.return_value = None
        mock_exists.side_effect = lambda x: False
        path = video_analyzer.get_executable_path('ffmpeg')
        self.assertIsNone(path)

    @patch('video_analyzer.requests.post')
    @patch('builtins.open', new_callable=mock_open, read_data=b'data')
    def test_upload_video_success(self, mock_file, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "file": {
                "name": "files/12345",
                "uri": "https://uri/12345"
            }
        }
        mock_post.return_value = mock_response

        result = video_analyzer.upload_video('test.mp4', 'dummy_key')
        self.assertIsNotNone(result)
        self.assertEqual(result['name'], 'files/12345')
        self.assertEqual(result['uri'], 'https://uri/12345')
        mock_post.assert_called_once()

    @patch('video_analyzer.requests.post')
    def test_generate_content_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "This is a test response"}
                        ]
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        # Test normal generation
        text = video_analyzer.generate_content('dummy_uri', 'question', 'dummy_key')
        self.assertEqual(text, "This is a test response")
        
        # Test JSON response extraction
        mock_response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": '```json\n{"answer": "test", "timestamps": [{"start": 1, "end": 2}]}\n```'}
                        ]
                    }
                }
            ]
        }
        text = video_analyzer.generate_content('dummy_uri', 'question', 'dummy_key', response_schema_json=True)
        self.assertEqual(text, '{"answer": "test", "timestamps": [{"start": 1, "end": 2}]}')

    @patch('video_analyzer.requests.get')
    def test_wait_for_file_active(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        
        # Test ACTIVE state
        mock_response.json.return_value = {"state": "ACTIVE"}
        mock_get.return_value = mock_response
        self.assertTrue(video_analyzer.wait_for_file_active('dummy_file', 'dummy_key', timeout=5))

        # Test FAILED state
        mock_response.json.return_value = {"state": "FAILED"}
        mock_get.return_value = mock_response
        self.assertFalse(video_analyzer.wait_for_file_active('dummy_file', 'dummy_key', timeout=5))

    @patch('video_analyzer.os.path.getsize')
    def test_truncate_video_to_limit_under_limit(self, mock_getsize):
        mock_getsize.return_value = 1000 # Under 2GB
        path, truncated = video_analyzer.truncate_video_to_limit('test.mp4')
        self.assertEqual(path, 'test.mp4')
        self.assertFalse(truncated)

if __name__ == '__main__':
    unittest.main()
