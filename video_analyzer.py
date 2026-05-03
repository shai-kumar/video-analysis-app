import argparse
import json
import os
import sys
import time
import requests

def upload_video(file_path, api_key):
    """Uploads a video to the Gemini File API using REST."""
    print(f"Uploading {file_path} to Gemini File API...")
    
    # Simple upload URL
    url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={api_key}"
    
    mime_type = "video/mp4"
    if file_path.endswith(".webm"):
        mime_type = "video/webm"
        
    headers = {
        "X-Goog-Upload-Protocol": "multipart",
        "X-Goog-Upload-Command": "upload, finalize",
    }
    
    metadata = {"file": {"displayName": os.path.basename(file_path)}}
    
    with open(file_path, "rb") as f:
        files = {
            "metadata": (None, json.dumps(metadata), "application/json"),
            "file": (os.path.basename(file_path), f, mime_type)
        }
        
        response = requests.post(url, headers=headers, files=files)
        
    if response.status_code != 200:
        print(f"Error uploading file: {response.text}")
        response.raise_for_status()
        
    result = response.json()
    file_info = result.get("file")
    if not file_info:
        print(f"Unexpected response format: {result}")
        return None
        
    print(f"Upload successful. File name: {file_info['name']}, URI: {file_info['uri']}")
    return file_info

def generate_content(file_uri, question, api_key, model_name="gemini-2.5-flash", response_schema_json=False):
    """Generates content based on the uploaded video and question."""
    print(f"Generating content with model {model_name}...")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    
    prompt_text = question
    if response_schema_json:
        prompt_text += ("\n\nYou MUST return a JSON object with exactly two keys:\n"
                       "1. 'answer': A string containing the answer to the user's question.\n"
                       "2. 'timestamps': A list of objects with 'start' and 'end' keys in seconds (numbers), e.g. [{'start': 12.5, 'end': 15.0}].")

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "video/mp4", # Assuming mp4 for the uploaded video
                            "fileUri": file_uri
                        }
                    },
                    {
                        "text": prompt_text
                    }
                ]
            }
        ]
    }
    
    if response_schema_json:
        payload["generationConfig"] = {"responseMimeType": "application/json"}
        
    response = requests.post(url, json=payload)
    if response.status_code != 200:
        print(f"Error generating content: {response.text}")
        response.raise_for_status()
        
    result = response.json()
    
    try:
        text = result['candidates'][0]['content']['parts'][0]['text']
        if response_schema_json:
            import re
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
        return text
    except (KeyError, IndexError) as e:
        print(f"Error parsing response: {e}")
        print(f"Raw response: {result}")
        return None

def delete_file(file_name, api_key):
    """Deletes the file from Gemini File API."""
    print(f"Cleaning up file {file_name}...")
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"
    response = requests.delete(url)
    if response.status_code == 200:
        print("Cleanup successful.")
    else:
        print(f"Error deleting file: {response.text}")

def wait_for_file_active(file_name, api_key, timeout=300):
    """Polls the file state until it is ACTIVE or FAILED."""
    print("Waiting for file to be processed...")
    url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={api_key}"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error checking file status: {response.text}")
            response.raise_for_status()
            
        result = response.json()
        state = result.get("state")
        if state == "ACTIVE":
            print("File is active.")
            return True
        elif state == "FAILED":
            print("File processing failed.")
            return False
        elif state == "PROCESSING":
            print(f"File state is PROCESSING. Waiting 5 seconds...")
        else:
            print(f"File state: {state}. Waiting 5 seconds...")
            
        time.sleep(5)
        
    print("Timeout waiting for file to be active.")
    return False

def clip_video(input_video_path, output_clip_path, timestamps, pre_padding=0.0, post_padding=0.0):
    """Clips a video according to timestamps and saves the combined output to output_clip_path."""
    import shutil
    import subprocess
    import tempfile

    if not timestamps:
        print("No timestamps found to extract clips.")
        return

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Error: 'ffmpeg' is not installed or not found in PATH. Clipping skipped.")
        return

    temp_dir = tempfile.mkdtemp()
    print(f"Extracting {len(timestamps)} segments to temporary directory: {temp_dir}")

    segment_files = []
    try:
        for i, segment in enumerate(timestamps):
            start = segment.get("start")
            end = segment.get("end")
            if start is None or end is None:
                print(f"Skipping invalid segment: {segment}")
                continue

            try:
                start_sec = float(start)
                end_sec = float(end)
            except ValueError:
                start_sec = float(str(start))
                end_sec = float(str(end))
                
            start_sec = max(0.0, start_sec - pre_padding)
            end_sec = end_sec + post_padding

            temp_output = os.path.join(temp_dir, f"segment_{i}.mp4")
            
            cmd = [
                ffmpeg_path, "-y",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", input_video_path,
                "-c", "copy",
                temp_output
            ]
            
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"ffmpeg error extracting segment {i}: {res.stderr.decode('utf-8', errors='ignore')}")
                continue
            
            if os.path.exists(temp_output):
                segment_files.append(temp_output)

        if not segment_files:
            print("No segments were successfully extracted.")
            return

        if len(segment_files) == 1:
            shutil.copy2(segment_files[0], output_clip_path)
            print(f"Successfully created single clip at {output_clip_path}")
        else:
            list_file_path = os.path.join(temp_dir, "concat_list.txt")
            with open(list_file_path, "w") as f:
                for seg_file in segment_files:
                    f.write(f"file '{os.path.abspath(seg_file)}'\n")

            cmd = [
                ffmpeg_path, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file_path,
                "-c", "copy",
                output_clip_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                print(f"ffmpeg error concatenating segments: {res.stderr.decode('utf-8', errors='ignore')}")
            else:
                print(f"Successfully created combined clip at {output_clip_path}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def truncate_video_to_limit(video_path, max_size_bytes=2000000000):
    """If the video file size exceeds max_size_bytes, trims it down so it's under the limit."""
    import shutil
    import subprocess
    import tempfile

    file_size = os.path.getsize(video_path)
    if file_size <= max_size_bytes:
        return video_path, False

    print(f"Original video size ({file_size} bytes) exceeds 2GB (2000000000 bytes) limit.")
    print("Truncating the video to stay within the 2GB limit before uploading...")

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Warning: ffmpeg not found. Cannot truncate video. Uploading as is.")
        return video_path, False

    ffprobe_path = shutil.which("ffprobe")
    duration = None
    if ffprobe_path:
        try:
            cmd = [
                ffprobe_path, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                video_path
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode == 0 and res.stdout.strip():
                duration = float(res.stdout.strip())
        except Exception as e:
            print(f"ffprobe warning: {e}")

    temp_truncated = os.path.join(tempfile.gettempdir(), f"truncated_{os.path.basename(video_path)}")

    if duration:
        # Use a safety margin to ensure the file size remains under the limit
        ratio = (max_size_bytes * 0.95) / file_size
        target_duration = duration * ratio
        print(f"Trimming duration to ~{target_duration:.2f} seconds out of {duration:.2f} seconds...")
        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-t", str(target_duration),
            "-c", "copy",
            temp_truncated
        ]
    else:
        print("Duration could not be determined. Truncating via file size limit (-fs)...")
        cmd = [
            ffmpeg_path, "-y",
            "-i", video_path,
            "-fs", str(max_size_bytes),
            "-c", "copy",
            temp_truncated
        ]

    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print(f"ffmpeg error truncating video: {res.stderr.decode('utf-8', errors='ignore')}")
            return video_path, False

        if os.path.exists(temp_truncated) and os.path.getsize(temp_truncated) > 0:
            print(f"Successfully truncated video. New size: {os.path.getsize(temp_truncated)} bytes")
            return temp_truncated, True
    except Exception as e:
        print(f"Error while executing ffmpeg truncation: {e}")

    return video_path, False

def analyze_video(video_path, question, model_name="gemini-2.5-flash", output_clip=None, pre_padding=0.0, post_padding=0.0):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    truncated_path, is_truncated = truncate_video_to_limit(video_path)

    file_info = None
    try:
        file_info = upload_video(truncated_path, api_key)
        if not file_info:
            return

        file_uri = file_info["uri"]
        file_name = file_info["name"]

        # Wait for file to be active before generating content
        if not wait_for_file_active(file_name, api_key):
            delete_file(file_name, api_key)
            return

        if output_clip:
            response_text = generate_content(file_uri, question, api_key, model_name, response_schema_json=True)
            if response_text:
                try:
                    data = json.loads(response_text)
                    answer = data.get("answer")
                    timestamps = data.get("timestamps", [])
                    if answer:
                        print("\n--- Answer ---")
                        print(answer)
                        print("--------------")
                    if timestamps:
                        print(f"\nFound {len(timestamps)} segment(s) matching the query. Extracting clip...")
                        clip_video(truncated_path, output_clip, timestamps, pre_padding=pre_padding, post_padding=post_padding)
                    else:
                        print("\nNo relevant scenes or timestamps were identified for clipping.")
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON response from Gemini: {e}")
                    print("Raw response received:")
                    print(response_text)
        else:
            answer = generate_content(file_uri, question, api_key, model_name, response_schema_json=False)
            if answer:
                print("\n--- Answer ---")
                print(answer)
                print("--------------")
    finally:
        if file_info and 'name' in file_info:
            delete_file(file_info['name'], api_key)
        if is_truncated and os.path.exists(truncated_path):
            try:
                os.remove(truncated_path)
            except OSError:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze a video and answer questions using Gemini.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--question", required=True, help="Question to ask about the video.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model to use.")
    parser.add_argument("--output-clip", help="Path to save the generated combined clip.")
    parser.add_argument("--pre-padding", type=float, default=0.0, help="Seconds to add before each clip timestamp.")
    parser.add_argument("--post-padding", type=float, default=0.0, help="Seconds to add after each clip timestamp.")
    
    args = parser.parse_args()
    
    analyze_video(args.video, args.question, args.model, args.output_clip, args.pre_padding, args.post_padding)
