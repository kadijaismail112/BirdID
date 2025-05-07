import os
from pydub import AudioSegment
import math

input_dir = "California Least Tern Bird Call"
output_dir = "California Least Tern Bird Call/processed_chunks"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".mp3"):
        input_path = os.path.join(input_dir, filename)
        
        audio = AudioSegment.from_mp3(input_path)
       
        duration_seconds = len(audio) / 1000.0
        
        # If audio is longer than 5 seconds, split it
        if duration_seconds > 5:
            # Calculate number of chunks
            num_chunks = math.ceil(duration_seconds / 5)
            
            # Split into chunks
            for i in range(num_chunks):
                start_time = i * 5000  # 5 seconds in milliseconds
                end_time = min((i + 1) * 5000, len(audio))
                
                chunk = audio[start_time:end_time]
                
                base_name = os.path.splitext(filename)[0]
                chunk_filename = f"{base_name}_chunk{i+1}.mp3"
                output_path = os.path.join(output_dir, chunk_filename)
                
                chunk.export(output_path, format="mp3")
                print(f"Created chunk {i+1} of {num_chunks} for {filename}")
        else:
            output_path = os.path.join(output_dir, filename)
            audio.export(output_path, format="mp3")
            print(f"Copied {filename} (duration: {duration_seconds:.2f} seconds)")
