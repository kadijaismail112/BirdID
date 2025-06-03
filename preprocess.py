import os
from pydub import AudioSegment
import math

input_folders = ["clt_bird_call", "other_bird_call"]

for base_folder in input_folders:
    for root, dirs, files in os.walk(base_folder):
        # Skip any processed_chunks folders
        if 'processed_chunks' in root:
            continue
        
        output_dir = os.path.join(root, "processed_chunks")
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nProcessing folder: {root}")
        
        for filename in files:
            if filename.endswith((".mp3", ".wav")):
                input_path = os.path.join(root, filename)
                # Skip if file is in processed_chunks directory
                if "processed_chunks" in input_path:
                    continue
                try:
                    # Load audio file
                    if filename.endswith(".mp3"):
                        audio = AudioSegment.from_mp3(input_path)
                    else:
                        audio = AudioSegment.from_wav(input_path)
                    # Set sample rate to 48kHz
                    audio = audio.set_frame_rate(48000)
                    duration_seconds = len(audio) / 1000.0
                    # If audio is longer than 3 seconds, split it
                    if duration_seconds > 3:
                        num_chunks = math.ceil(duration_seconds / 3)
                        for i in range(num_chunks):
                            start_time = i * 3000  # 3 seconds in ms
                            end_time = min((i + 1) * 3000, len(audio))
                            chunk = audio[start_time:end_time]
                            base_name = os.path.splitext(filename)[0]
                            chunk_filename = f"{base_name}_chunk{i+1}.wav"
                            output_path = os.path.join(output_dir, chunk_filename)
                            chunk.export(output_path, format="wav")
                            print(f"Created chunk {i+1} of {num_chunks} for {filename} in {root}")
                    else:
                        base_name = os.path.splitext(filename)[0]
                        output_path = os.path.join(output_dir, f"{base_name}.wav")
                        audio.export(output_path, format="wav")
                        print(f"Converted {filename} to WAV (duration: {duration_seconds:.2f} seconds) in {root}")
                except Exception as e:
                    print(f"Error processing {filename} in {root}: {str(e)}")
                    continue
