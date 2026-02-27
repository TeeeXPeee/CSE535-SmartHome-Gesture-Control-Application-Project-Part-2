import os
import subprocess
import shutil
import tempfile

def clean_videos_in_dir(video_dir):
    for filename in os.listdir(video_dir):
        if not filename.lower().endswith(".mp4"):
            continue

        input_path = os.path.join(video_dir, filename)

        # Create temp output file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        os.close(temp_fd)  # ffmpeg will write to this

        print(f"🔧 Cleaning: {filename}")

        cmd = [
            "ffmpeg",
            "-y",                      # overwrite temp file if exists
            "-i", input_path,
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            temp_path
        ]

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if result.returncode != 0:
                print(f"❌ FFmpeg failed on {filename}")
                print(result.stderr)
                os.remove(temp_path)
                continue

            # Replace original file safely
            backup_path = input_path + ".bak"
            os.replace(input_path, backup_path)   # atomic move
            os.replace(temp_path, input_path)     # new file takes original name
            os.remove(backup_path)                # delete old file

            print(f"✅ Replaced: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == "__main__":
    clean_videos_in_dir("traindata")