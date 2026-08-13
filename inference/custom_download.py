import os
import time
import requests

url = "https://huggingface.co/Systran/faster-whisper-medium/resolve/main/model.bin"
filepath = "models/faster-whisper-medium/model.bin"
os.makedirs(os.path.dirname(filepath), exist_ok=True)

def download_with_resume():
    max_retries = 20
    for attempt in range(max_retries):
        try:
            downloaded = 0
            if os.path.exists(filepath):
                downloaded = os.path.getsize(filepath)
            
            headers = {}
            if downloaded > 0:
                headers['Range'] = f'bytes={downloaded}-'
                
            print(f"Attempt {attempt+1}: Resuming from {downloaded} bytes...")
            
            response = requests.get(url, headers=headers, stream=True, timeout=10)
            
            if response.status_code == 416: # Range Not Satisfiable (already fully downloaded)
                print("Download complete (416)!")
                return True
                
            if response.status_code not in (200, 206):
                print(f"Failed with status code {response.status_code}")
                time.sleep(2)
                continue

            mode = 'ab' if response.status_code == 206 else 'wb'
            with open(filepath, mode) as f:
                for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        
            print("Download complete!")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            time.sleep(2)
    return False

if __name__ == "__main__":
    download_with_resume()
