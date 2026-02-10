import pandas as pd
import pretty_midi
import os
import soundfile as sf
import tempfile
from utils.config import SETTINGS

class OmnizartWrapper:
    def __init__(self):
        # Omnizart is imported inside methods to avoid heavy imports if not used
        try:
            from omnizart.drum import app
            self.app = app
            print("Omnizart loaded.")
        except ImportError:
            print("Omnizart not found. Please install it with `pip install omnizart`.")
            self.app = None

    def predict(self, samples, sr):
        """
        Predict drum hits from audio samples.
        Args:
            samples (np.ndarray): Audio samples.
            sr (int): Sampling rate.
        Returns:
            pd.DataFrame: DataFrame with columns ['time', 'prediction', 'confidence']
        """
        if self.app is None:
            raise ImportError("Omnizart not installed.")

        # Omnizart expects a file path, not raw samples.
        # We need to save samples to a temporary wav file.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_path = temp_audio.name
            
        try:
            sf.write(temp_path, samples, sr, subtype='PCM_16')
            
            # Omnizart transcription
            midi_data = self.app.transcribe(temp_path)
            
        except Exception as e:
            print(f"Error in Omnizart transcription: {e}")
            return pd.DataFrame(columns=['time', 'prediction', 'confidence'])
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Process MIDI data
        # Mapping MIDI pitches to our labels
        label_map = {
            35: 'kick_drum', 36: 'kick_drum',
            38: 'snare', 40: 'snare',
            37: 'snare', # Side Stick
            42: 'hihat_c', 44: 'hihat_c', 46: 'hihat_c', # Simplified all HH to closed for now or map appropriately
            41: 'tom_h', 43: 'tom_h', 45: 'tom_h', 47: 'tom_h', 48: 'tom_h', 50: 'tom_h',
            49: 'crash', 57: 'crash',
            51: 'ride', 59: 'ride'
        }

        predictions = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                continue
                
            for note in instrument.notes:
                pitch = note.pitch
                time = note.start
                velocity = note.velocity / 127.0 # precise confidence
                
                if pitch in label_map:
                    label = label_map[pitch]
                    predictions.append({
                        'time': time,
                        'prediction': label,
                        'confidence': velocity
                    })
        
        df = pd.DataFrame(predictions)
        if df.empty:
            return pd.DataFrame(columns=['time', 'prediction', 'confidence'])
            
        return df

if __name__ == "__main__":
    # Test stub
    pass
