import os
import subprocess
import shutil

class DemucsSeparator:
    def __init__(self, output_dir="separated"):
        self.output_dir = output_dir
        # Ensure output directory exists (demucs creates it, but good to know)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def separate(self, audio_path):
        """
        Separates the audio file using Demucs and returns the path to the drums.wav.
        """
        print(f"Separating audio: {audio_path}")
        
        # Run Demucs CLI
        # -n htdemucs: Use the high-performance Hybrid Transformer model
        # --two-stems=drums: Only separate drums (faster)
        # -o: Output directory
        command = [
            "demucs",
            "-n", "htdemucs",
            "--two-stems=drums",
            "-o", self.output_dir,
            audio_path
        ]
        
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running Demucs: {e}")
            return None
        except FileNotFoundError:
             print("Demucs command not found. Please install with `pip install demucs`.")
             return None

        # Construct path to the output file
        # Demucs output structure: output_dir/htdemucs/filename_no_ext/drums.wav
        filename = os.path.basename(audio_path)
        filename_no_ext = os.path.splitext(filename)[0]
        
        drums_path = os.path.join(self.output_dir, "htdemucs", filename_no_ext, "drums.wav")
        
        if os.path.exists(drums_path):
            print(f"Separation complete. Drums at: {drums_path}")
            return drums_path
        else:
            print(f"Separation failed. Could not find output file: {drums_path}")
            return None

if __name__ == "__main__":
    # Test stub
    pass
