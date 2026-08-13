import time
from huggingface_hub import snapshot_download

max_retries = 10
for i in range(max_retries):
    try:
        print(f"Attempt {i+1} to download faster-whisper-medium...")
        snapshot_download('Systran/faster-whisper-medium', local_dir='models/faster-whisper-medium')
        print("Download complete!")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)
