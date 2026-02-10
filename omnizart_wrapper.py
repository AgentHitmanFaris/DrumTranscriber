import pandas as pd
import pretty_midi
import os
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

    def predict(self, audio_path):
        if self.app is None:
            raise ImportError("Omnizart not installed.")

        # Omnizart transcription
        # It usually outputs a MIDI file to the same directory or specified output
        try:
            midi_data = self.app.transcribe(audio_path)
            # The transcribe function might return a MIDI object directly or write file
            # Checking documentation/source behavior: usually writes file. 
            # But let's check if we can get the MIDI object directly.
            # If not, we reload the midi file.
        except Exception as e:
            print(f"Error in Omnizart transcription: {e}")
            return pd.DataFrame(columns=['time', 'prediction', 'confidence'])

        # Process MIDI data
        # Mapping MIDI pitches to our labels
        # GM Drum Map: 
        # 35, 36 -> Kick
        # 38, 40 -> Snare
        # 42, 44, 46 -> HiHat (Closed/Pedal/Open)
        # 41, 43, 45, 47, 48, 50 -> Toms
        # 49, 57 -> Crash
        # 51, 59 -> Ride
        
        # We need to standardize this mapping.
        # Reverse map from SETTINGS['LABELS_INDEX'] which is:
        # {0: 'kick_drum', 1: 'snare', ...}
        
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
