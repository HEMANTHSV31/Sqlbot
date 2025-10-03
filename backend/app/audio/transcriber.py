import io
import wave
import numpy as np
from faster_whisper import WhisperModel
import noisereduce as nr
from typing import Tuple, Optional
import aiofiles
import tempfile
import os
import librosa

class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )
    
    async def transcribe_audio(self, audio_file) -> Tuple[str, float]:
        """Transcribe audio file to English text"""
        # Save uploaded file temporarily
        async with aiofiles.tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio_file.read()
            await tmp.write(content)
            temp_path = tmp.name
        
        try:
            # Convert to WAV if needed and load audio
            audio_data, sample_rate = await self._load_and_convert_audio(temp_path)
            
            # Apply noise reduction
            cleaned_audio = nr.reduce_noise(y=audio_data, sr=sample_rate)
            
            # Save cleaned audio to temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as clean_file:
                self._save_wav(clean_file.name, cleaned_audio, sample_rate)
                
                # Transcribe with language detection and translation to English
                segments, info = self.model.transcribe(
                    clean_file.name,
                    language=None,  # auto-detect
                    task="translate",  # translate to English
                    beam_size=5
                )
                
                # Combine segments
                transcript = " ".join(segment.text for segment in segments)
                confidence = np.mean([segment.avg_logprob for segment in segments]) if segments else 0.0
                
                os.unlink(clean_file.name)
                return transcript.strip(), float(confidence)
        
        finally:
            os.unlink(temp_path)
    
    async def _load_and_convert_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load and convert audio file to numpy array"""
        try:
            # Use librosa to load audio with resampling to 16kHz
            audio, sr = librosa.load(file_path, sr=16000, mono=True)
            return audio, sr
        except Exception as e:
            raise Exception(f"Audio loading failed: {str(e)}")
    
    def _save_wav(self, filename: str, audio: np.ndarray, sample_rate: int):
        """Save numpy array as WAV file"""
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())