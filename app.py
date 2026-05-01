import streamlit as st
import tempfile
import os
import json
import sys

# Ensure video_analyzer is accessible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_analyzer import upload_video, wait_for_file_active, generate_content, clip_video, delete_file, truncate_video_to_limit

st.set_page_config(page_title="Video Analysis Tool", layout="centered")

st.title("Video Analysis & Clipping App")
st.write("Upload a video, describe the events you want to find, and automatically extract a combined clip of those moments.")

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("GEMINI_API_KEY environment variable is not set. Please set it before running the app.")
    st.stop()

# UI Elements
uploaded_file = st.file_uploader("Upload Video", type=["mp4", "webm", "mov", "avi"])

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

if st.button("Analyze & Extract", type="primary"):
    if not uploaded_file:
        st.warning("Please upload a video file first.")
        st.stop()
    if not query:
        st.warning("Please enter a description of the events to find.")
        st.stop()

    with st.spinner("Saving uploaded video..."):
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tf:
            tf.write(uploaded_file.read())
            temp_video_path = tf.name

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
                delete_file(file_name, api_key)
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
            
            # Clean up from Gemini API
            delete_file(file_name, api_key)

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
                    output_clip_path = tempfile.mktemp(suffix=".mp4")
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
                        
                        # Optionally display the video in the app
                        st.video(clip_bytes)
                    else:
                        st.error("Failed to generate the final clip.")

        except json.JSONDecodeError as e:
            st.error(f"Failed to parse response from Gemini: {e}")
            with st.expander("Raw Response"):
                st.text(response_text)

    finally:
        # Clean up local files
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        if 'truncated_path' in locals() and is_truncated and os.path.exists(truncated_path):
            try:
                os.remove(truncated_path)
            except OSError:
                pass
        if 'output_clip_path' in locals() and os.path.exists(output_clip_path):
            try:
                os.remove(output_clip_path)
            except OSError:
                pass
