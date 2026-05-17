import streamlit as st
import tempfile
import os
import json
import sys

# Ensure video_analyzer is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_analyzer import upload_video, wait_for_file_active, generate_content, clip_video, delete_file, truncate_video_to_limit
import drive_utils

st.set_page_config(page_title="Video Analysis Tool", layout="centered")

st.title("Video Analysis & Clipping App")
st.write("Upload a video, describe the events you want to find, and automatically extract a combined clip of those moments.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

# UI Elements
st.subheader("1. Select Video Source")
video_source = st.radio("Choose how to provide your video:", 
                        ["Upload Local File", "Public Google Drive Link", "My Google Drive"])

uploaded_file = None
public_drive_link = None
selected_drive_file_id = None
drive_service = None

if video_source == "Upload Local File":
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "webm", "mov", "avi"])

elif video_source == "Public Google Drive Link":
    public_drive_link = st.text_input("Paste Public Google Drive Link (Make sure anyone with the link can view)")

elif video_source == "My Google Drive":
    st.info("To use this feature, you must have a `credentials.json` file in the project folder from Google Cloud Console.")
    try:
        drive_service = drive_utils.get_drive_service()
        if not drive_service:
            st.error("Could not find `credentials.json`. Please place it in the same directory as this app.")
        else:
            with st.spinner("Fetching your videos..."):
                videos = drive_utils.list_drive_videos(drive_service)
                if not videos:
                    st.warning("No video files found in your Google Drive.")
                else:
                    video_options = {v['name']: v['id'] for v in videos}
                    selected_video_name = st.selectbox("Select a video from your Drive:", list(video_options.keys()))
                    selected_drive_file_id = video_options[selected_video_name]
    except Exception as e:
        st.error(f"Error authenticating with Google Drive: {e}")

st.subheader("2. Analysis Settings")
query = st.text_area("Event Description", placeholder="e.g., Identify the player serving the ball")

col1, col2, col3 = st.columns(3)
with col1:
    num_events = st.number_input("Number of events", min_value=1, value=3, step=1, 
                                 help="Maximum number of events to find and extract.")
with col2:
    pre_padding = st.number_input("Pre-padding (s)", min_value=0.0, value=1.0, step=0.5,
                                  help="Seconds to include before the event starts.")
with col3:
    post_padding = st.number_input("Post-padding (s)", min_value=0.0, value=1.0, step=0.5,
                                   help="Seconds to include after the event ends.")

save_to_drive = False
if video_source == "My Google Drive" and drive_service:
    save_to_drive = st.checkbox("Save final clip back to my Google Drive")

st.subheader("3. Processing Options")
process_mode = st.radio("Notification Method:", ["Wait for processing to finish here", "Email me when it's done (Background Task)"])
recipient_email = None
if process_mode == "Email me when it's done (Background Task)":
    recipient_email = st.text_input("Email address to notify:")
    st.info("Note: The host machine must have SMTP_EMAIL and SMTP_PASSWORD environment variables set to send the email.")

if st.button("Analyze & Extract", type="primary"):
    if video_source == "Upload Local File" and not uploaded_file:
        st.warning("Please upload a video file first.")
        st.stop()
    elif video_source == "Public Google Drive Link" and not public_drive_link:
        st.warning("Please enter a Google Drive link.")
        st.stop()
    elif video_source == "My Google Drive" and not selected_drive_file_id:
        st.warning("Please select a video from your Drive.")
        st.stop()
        
    if not query:
        st.warning("Please enter a description of the events to find.")
        st.stop()

    if process_mode == "Email me when it's done (Background Task)" and not recipient_email:
        st.warning("Please enter an email address for notification.")
        st.stop()

    with st.spinner("Preparing video..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
            temp_video_path = tf.name
            
        if video_source == "Upload Local File":
            with open(temp_video_path, 'wb') as f:
                f.write(uploaded_file.read())

    if process_mode == "Email me when it's done (Background Task)":
        from background_task import run_background_analysis
        run_background_analysis(
            video_source, public_drive_link, selected_drive_file_id, 
            query, num_events, pre_padding, post_padding, save_to_drive, 
            api_key, temp_video_path, recipient_email
        )
        st.success("Background task started successfully! You can safely close this page. You will receive an email when your video is ready.")
        st.stop()

    if video_source == "Public Google Drive Link":
        with st.spinner("Downloading from public link..."):
            res = drive_utils.download_public_link(public_drive_link, temp_video_path)
            if not res:
                st.error("Failed to download from public link.")
                st.stop()
    elif video_source == "My Google Drive":
        with st.spinner("Downloading from your Google Drive..."):
            res = drive_utils.download_file_from_drive(drive_service, selected_drive_file_id, temp_video_path)
            if not res:
                st.error("Failed to download from Google Drive.")
                st.stop()

    truncated_path = None
    is_truncated = False
    output_clip_path = None
    file_name = None

    try:
        with st.spinner("Checking video size..."):
            truncated_path, is_truncated = truncate_video_to_limit(temp_video_path)

        with st.spinner("Uploading video to Gemini API..."):
            file_info = upload_video(truncated_path, api_key)
            if not file_info:
                st.error("Failed to upload video.")
                st.stop()
                
            file_uri = file_info["uri"]
            file_name = file_info["name"]

        with st.spinner("Waiting for video processing (this may take a minute)..."):
            is_active = wait_for_file_active(file_name, api_key)
            if not is_active:
                st.error("Video processing failed or timed out.")
                st.stop()

        with st.spinner("Analyzing video content..."):
            # Modify the query to incorporate num_events and video description
            full_query = (f"{query}\n\n"
                          f"Instructions:\n"
                          f"- Identify up to {num_events} occurrences of the specified event.\n"
                          f"- Provide a general description of the entire video content in the answer.")
            
            response_text = generate_content(
                file_uri, 
                full_query, 
                api_key, 
                model_name="gemini-2.5-flash", 
                response_schema_json=True
            )

        if not response_text:
            st.error("No response received from the model.")
            st.stop()

        # Parse JSON response
        try:
            data = json.loads(response_text)
            answer = data.get("answer", "No description provided.")
            timestamps = data.get("timestamps", [])
            
            # Limit timestamps to num_events if the model returned more
            timestamps = timestamps[:int(num_events)]

            # Display the video description
            st.subheader("Video Description & Analysis")
            st.write(answer)

            if not timestamps:
                st.warning("No events found matching the description.")
            else:
                st.success(f"Found {len(timestamps)} event(s).")
                st.json(timestamps)
                
                with st.spinner(f"Extracting and combining {len(timestamps)} clip(s)..."):
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                        output_clip_path = tf.name
                    st.success("Created temp file.")
                    clip_video(
                        truncated_path, 
                        output_clip_path, 
                        timestamps, 
                        pre_padding=pre_padding, 
                        post_padding=post_padding
                    )
                    
                    if os.path.exists(output_clip_path):
                        st.success("Clip extraction complete! Ready for download.")
                        
                        with open(output_clip_path, "rb") as f:
                            clip_bytes = f.read()
                            
                        st.download_button(
                            label="Download Extracted Clip (MP4)",
                            data=clip_bytes,
                            file_name="extracted_clip.mp4",
                            mime="video/mp4"
                        )
                        
                        st.video(clip_bytes)

                        if save_to_drive and drive_service:
                            with st.spinner("Uploading final clip to Google Drive..."):
                                file_id = drive_utils.upload_file_to_drive(drive_service, output_clip_path, "extracted_clip.mp4")
                                if file_id:
                                    st.success(f"Successfully saved to your Google Drive! (File ID: {file_id})")
                                else:
                                    st.error("Failed to save to Google Drive.")
                    else:
                        st.error("Failed to generate the final clip.")

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse response from Gemini: {e}")
            with st.expander("Raw Response"):
                st.text(response_text)

    finally:
        # Clean up Gemini API file
        if file_name:
            delete_file(file_name, api_key)

        # Clean up local files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if is_truncated and truncated_path and os.path.exists(truncated_path):
            try:
                os.remove(truncated_path)
            except OSError:
                pass
        if output_clip_path and os.path.exists(output_clip_path):
            try:
                os.remove(output_clip_path)
            except OSError:
                pass
