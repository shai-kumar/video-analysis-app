import threading
import tempfile
import os
import json
from video_analyzer import upload_video, wait_for_file_active, generate_content, clip_video, delete_file, truncate_video_to_limit
import drive_utils
import smtplib
from email.message import EmailMessage

def send_notification_email(recipient_email, subject, body, attachment_path=None):
    sender_email = os.environ.get("SMTP_EMAIL")
    sender_password = os.environ.get("SMTP_PASSWORD")
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", 587))

    if not sender_email or not sender_password:
        print("SMTP credentials not configured. Cannot send email.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg.set_content(body)

    if attachment_path and os.path.exists(attachment_path):
        import mimetypes
        ctype, encoding = mimetypes.guess_type(attachment_path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        with open(attachment_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(attachment_path))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_background_analysis(video_source, public_drive_link, selected_drive_file_id, 
                            query, num_events, pre_padding, post_padding, save_to_drive, 
                            api_key, temp_video_path, recipient_email):
    def task():
        truncated_path = None
        is_truncated = False
        output_clip_path = None
        file_name = None
        drive_service = drive_utils.get_drive_service() if save_to_drive or video_source == "My Google Drive" else None

        try:
            # Download video if needed
            if video_source == "Public Google Drive Link":
                res = drive_utils.download_public_link(public_drive_link, temp_video_path)
                if not res: return
            elif video_source == "My Google Drive" and drive_service:
                res = drive_utils.download_file_from_drive(drive_service, selected_drive_file_id, temp_video_path)
                if not res: return

            truncated_path, is_truncated = truncate_video_to_limit(temp_video_path)

            file_info = upload_video(truncated_path, api_key)
            if not file_info: return
            
            file_uri = file_info["uri"]
            file_name = file_info["name"]

            if not wait_for_file_active(file_name, api_key): return

            full_query = (f"{query}\n\n"
                          f"Instructions:\n"
                          f"- Identify up to {num_events} occurrences of the specified event.\n"
                          f"- Provide a general description of the entire video content in the answer.")
            
            response_text = generate_content(
                file_uri, full_query, api_key, model_name="gemini-2.5-flash", response_schema_json=True
            )

            if not response_text: return

            data = json.loads(response_text)
            timestamps = data.get("timestamps", [])[:int(num_events)]
            answer = data.get("answer", "")
            
            if timestamps:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                    output_clip_path = tf.name
                
                clip_video(truncated_path, output_clip_path, timestamps, pre_padding=pre_padding, post_padding=post_padding)
                
                if os.path.exists(output_clip_path):
                    body = f"Your video processing is complete!\n\nAnalysis:\n{answer}\n\nFound {len(timestamps)} events."
                    
                    if save_to_drive and drive_service:
                        file_id = drive_utils.upload_file_to_drive(drive_service, output_clip_path, "extracted_clip.mp4")
                        if file_id:
                            body += f"\n\nSaved to Google Drive with File ID: {file_id}"
                            
                    send_notification_email(recipient_email, "Video Analysis Complete", body, attachment_path=output_clip_path)
            else:
                body = f"Your video processing is complete!\n\nAnalysis:\n{answer}\n\nNo events matching your description were found."
                send_notification_email(recipient_email, "Video Analysis Complete", body)

        except Exception as e:
            print(f"Background task failed: {e}")
            send_notification_email(recipient_email, "Video Analysis Failed", f"An error occurred: {e}")
            
        finally:
            if file_name:
                delete_file(file_name, api_key)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            if is_truncated and truncated_path and os.path.exists(truncated_path):
                try: os.remove(truncated_path)
                except: pass
            if output_clip_path and os.path.exists(output_clip_path):
                try: os.remove(output_clip_path)
                except: pass

    thread = threading.Thread(target=task)
    thread.daemon = True
    thread.start()
