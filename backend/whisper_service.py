import os
import tempfile
import uuid
import logging
import time
import shutil
import io
import wave
import numpy as np

from typing import Optional, Tuple

# --- Optional Dependencies ---
# These are imported in a try-except block so the application can start even if they are missing,
# although their corresponding features will be disabled.

try:
    # PyTorch is used to detect GPU availability.
    import torch
except ImportError:
    torch = None

try:
    # faster-whisper is the core transcription engine. It's a required dependency.
    from faster_whisper import WhisperModel
except ImportError as e:
    raise RuntimeError(
        "faster-whisper is a required dependency. Please install it with: pip install faster-whisper"
    ) from e

try:
    # noisereduce provides a lightweight way to clean up audio before transcription.
    import noisereduce as nr
except ImportError:
    nr = None  # Service will work without it, but denoising will be skipped.

# --- Setup Logging ---
logger = logging.getLogger(__name__)


class WhisperSpeechToText:
    """
    A service class to handle Whisper model loading and audio transcription using faster-whisper.
    This class is responsible for all speech-to-text operations, including device detection,
    model loading, and processing audio files.
    """

    def __init__(self, model_size: str = "base"):
        """
        Initializes the service by loading the specified Whisper model.

        Args:
            model_size (str): The model size to use (e.g., "tiny", "base", "small", "medium", "large-v3").
                               "base" is a good default for balancing speed and accuracy.
        """
        self.model_size = model_size

        # Detect the best available device (CUDA GPU or CPU) and compute precision.
        device, compute_type = self._detect_device_and_precision()
        logger.info(f"Loading faster-whisper model '{self.model_size}' on device '{device}' with compute type '{compute_type}'...")

        t0 = time.time()
        try:
            # Load the transcription model from Hugging Face Hub or local cache.
            self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            logger.info(f"Whisper model loaded successfully in {time.time() - t0:.2f}s")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_size}'. Error: {e}")
            raise RuntimeError(f"Could not initialize WhisperModel: {e}") from e

    def _detect_device_and_precision(self) -> Tuple[str, str]:
        """
        Detects the optimal device (CUDA or CPU) and compute type for transcription.
        - Prefers CUDA with float16 precision if a compatible GPU is found.
        - Falls back to CPU with int8 precision for broader compatibility and good performance.

        Returns:
            A tuple containing the device string ("cuda" or "cpu") and compute type string.
        """
        if torch is not None and torch.cuda.is_available():
            logger.info("CUDA GPU detected. Using 'cuda' device with 'float16' precision.")
            return "cuda", "float16"
        
        logger.info("No CUDA GPU detected. Falling back to 'cpu' device with 'int8' precision.")
        return "cpu", "int8"

    def _decode_and_resample_wav(self, audio_content: bytes) -> np.ndarray:
        """
        Decodes WAV audio content into a mono, 16kHz float32 NumPy array.
        This is a fallback for when FFmpeg isn't available and ensures WAV files can still be processed.

        Args:
            audio_content: The byte content of the WAV file.

        Returns:
            A NumPy array of the audio waveform, normalized to [-1.0, 1.0].
        
        Raises:
            ValueError: If the WAV format is not 16-bit PCM.
        """
        with io.BytesIO(audio_content) as buffer:
            with wave.open(buffer, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

        if sample_width != 2: # 16-bit PCM
            raise ValueError(f"Unsupported WAV sample width: {sample_width*8}-bit. Only 16-bit PCM is supported.")

        # Convert byte data to int16 NumPy array
        audio_int16 = np.frombuffer(frames, dtype=np.int16)

        # If stereo, convert to mono by averaging channels
        if n_channels > 1:
            audio_int16 = audio_int16.reshape(-1, n_channels).mean(axis=1).astype(np.int16)
        
        # Normalize to float32 in the range [-1.0, 1.0]
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Resample to 16kHz if necessary (Whisper's required sample rate)
        if sample_rate != 16000:
            resampling_factor = 16000 / sample_rate
            new_length = int(len(audio_float32) * resampling_factor)
            original_indices = np.linspace(0, len(audio_float32) - 1, len(audio_float32))
            new_indices = np.linspace(0, len(audio_float32) - 1, new_length)
            audio_float32 = np.interp(new_indices, original_indices, audio_float32).astype(np.float32)

        return audio_float32

    def _apply_noise_reduction(self, audio_waveform: np.ndarray) -> np.ndarray:
        """
        Applies spectral noise reduction to the audio waveform if the 'noisereduce'
        library is installed.

        Args:
            audio_waveform: A float32 NumPy array of the audio.

        Returns:
            The processed (or original, if library is missing) audio waveform.
        """
        if nr is None:
            return audio_waveform # Skip if library is not installed
        try:
            # The library expects a specific sample rate, which is 16000 after our resampling
            reduced_noise_audio = nr.reduce_noise(y=audio_waveform, sr=16000)
            return reduced_noise_audio
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}. Proceeding with original audio.")
            return audio_waveform

    def transcribe_audio_file(self, file_content: bytes, filename: str) -> dict:
        """
        Transcribes audio content from a byte stream using the loaded Whisper model.
        This method handles temporary file creation, audio decoding, and transcription.

        Args:
            file_content: The raw byte content of the audio file.
            filename: The original filename, used to infer the file extension.

        Returns:
            A dictionary containing the transcription text and language info.
            Example: {"text": "Show me top products sold today.", "language": "en"}

        Raises:
            Exception: If an error occurs during file processing or transcription.
        """
        temp_file_path: Optional[str] = None
        try:
            # faster-whisper works best with file paths, so we write the content to a temporary file.
            file_extension = os.path.splitext(filename)[1] or ".tmp"
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name

            logger.info(f"Temporary audio file created at: {temp_file_path}")

            # Define transcription options for better accuracy and speed.
            # VAD (Voice Activity Detection) helps filter out silence.
            transcription_options = {
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
            }

            # The transcribe method handles decoding from various formats if FFmpeg is installed.
            segments, info = self.model.transcribe(temp_file_path, **transcription_options)

            # Concatenate all transcribed segments into a single string.
            full_transcript = "".join(segment.text for segment in segments).strip()

            detected_language = info.language
            lang_probability = info.language_probability

            logger.info(
                f"Transcription complete. Language: {detected_language} "
                f"(Confidence: {lang_probability:.2f}). Text: '{full_transcript}'"
            )

            return {
                "text": full_transcript,
                "language": detected_language,
                "language_probability": lang_probability,
            }
        except Exception as e:
            logger.error(f"Error during transcription process: {e}", exc_info=True)
            raise  # Re-raise the exception to be handled by the API endpoint
        finally:
            # Ensure the temporary file is always cleaned up.
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except OSError as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")

# --- Direct Execution Block for Testing ---
# This block runs only when you execute the script directly (e.g., `python whisper_service.py`).
# It's useful for verifying that the model loads correctly without running the full FastAPI server.
if __name__ == "__main__":
    print("--- Testing WhisperSpeechToText Service ---")
    
    # Check if FFmpeg is available in the system's PATH
    if shutil.which("ffmpeg"):
        print("[INFO] FFmpeg is installed and accessible.")
    else:
        print("[WARNING] FFmpeg not found. Transcription will be limited to WAV files.")
        
    try:
        # Initialize the service with a small, fast model for testing.
        print("Initializing with model 'tiny'...")
        stt_service = WhisperSpeechToText(model_size="tiny")
        print("\n[SUCCESS] WhisperSpeechToText service initialized successfully.")
        print(f"  - Model Size: {stt_service.model_size}")
        
        # To perform a full transcription test, you would need a sample audio file.
        # Example:
        # try:
        #     with open("path/to/your/sample_audio.wav", "rb") as audio_file:
        #         content = audio_file.read()
        #         result = stt_service.transcribe_audio_file(content, "sample_audio.wav")
        #         print(f"\n--- Test Transcription Result ---")
        #         print(f"Detected Text: {result['text']}")
        #         print("---------------------------------")
        # except FileNotFoundError:
        #     print("\n[INFO] To run a full transcription test, provide a valid audio file path.")

    except Exception as e:
        print(f"\n[ERROR] An error occurred during the test: {e}")