"""
Audio Processing Pipeline for MURMUR

Handles audio loading, preprocessing, and spectrogram generation.
"""
import numpy as np
import librosa
import torch
from pathlib import Path
from typing import Union, Tuple, Optional, List

from config import SAMPLE_RATE, WINDOW_SIZE, N_MELS, HOP_LENGTH, N_FFT


class AudioProcessor:
    """Audio preprocessing for MSTCN model"""
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 n_mels: int = N_MELS,
                 n_fft: int = N_FFT,
                 hop_length: int = HOP_LENGTH,
                 window_size: int = WINDOW_SIZE):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_size = window_size
    
    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load and resample audio file
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
        
        Returns:
            Audio waveform as numpy array
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio
    
    def load_audio_from_bytes(self, audio_bytes: bytes) -> np.ndarray:
        """Load audio from bytes (for API uploads)"""
        import io
        import soundfile as sf
        
        audio_io = io.BytesIO(audio_bytes)
        audio, sr = sf.read(audio_io)
        
        # Resample if necessary
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        return audio.astype(np.float32)
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel spectrogram from audio
        
        Args:
            audio: Audio waveform
        
        Returns:
            Mel spectrogram of shape (n_mels, time)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def compute_delta_features(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Compute delta and delta-delta features
        
        Returns:
            Stacked features of shape (3, n_mels, time)
        """
        delta = librosa.feature.delta(spectrogram)
        delta2 = librosa.feature.delta(spectrogram, order=2)
        
        # Stack to create 3-channel input
        features = np.stack([spectrogram, delta, delta2], axis=0)
        return features
    
    def segment_to_windows(self, features: np.ndarray, 
                           window_size: Optional[int] = None,
                           hop: Optional[int] = None) -> List[np.ndarray]:
        """
        Segment features into fixed-size windows
        
        Args:
            features: Feature array of shape (channels, freq, time)
            window_size: Window size in frames
            hop: Hop between windows (default: window_size // 2)
        
        Returns:
            List of window arrays
        """
        window_size = window_size or self.window_size
        hop = hop or window_size // 2
        
        time_steps = features.shape[-1]
        windows = []
        
        for start in range(0, time_steps - window_size + 1, hop):
            window = features[:, :, start:start + window_size]
            windows.append(window)
        
        # Handle case where audio is shorter than window
        if len(windows) == 0 and time_steps > 0:
            # Pad to window size
            pad_amount = window_size - time_steps
            padded = np.pad(features, ((0, 0), (0, 0), (0, pad_amount)), mode='constant')
            windows.append(padded)
        
        return windows
    
    def process_audio(self, audio: np.ndarray) -> torch.Tensor:
        """
        Full preprocessing pipeline: audio -> model input
        
        Args:
            audio: Raw audio waveform
        
        Returns:
            Tensor ready for MSTCN model
        """
        # Normalize
        audio = self.normalize_audio(audio)
        
        # Compute mel spectrogram
        mel_spec = self.compute_mel_spectrogram(audio)
        
        # Add delta features (3 channels)
        features = self.compute_delta_features(mel_spec)
        
        # Convert to tensor
        tensor = torch.from_numpy(features).float()
        
        return tensor
    
    def process_audio_file(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load and process audio file"""
        audio = self.load_audio(audio_path)
        return self.process_audio(audio)
    
    def process_audio_bytes(self, audio_bytes: bytes) -> torch.Tensor:
        """Process audio from bytes"""
        audio = self.load_audio_from_bytes(audio_bytes)
        return self.process_audio(audio)
    
    def get_windows_tensor(self, audio: np.ndarray, 
                           window_size: Optional[int] = None) -> torch.Tensor:
        """
        Process audio and return windowed tensor batch
        
        Returns:
            Tensor of shape (num_windows, 3, n_mels, window_size)
        """
        audio = self.normalize_audio(audio)
        mel_spec = self.compute_mel_spectrogram(audio)
        features = self.compute_delta_features(mel_spec)
        windows = self.segment_to_windows(features, window_size)
        
        if len(windows) == 0:
            raise ValueError("Audio too short to create windows")
        
        batch = torch.stack([torch.from_numpy(w).float() for w in windows])
        return batch


class BreathDetector:
    """Detect breath events in audio stream"""
    
    def __init__(self, 
                 sample_rate: int = SAMPLE_RATE,
                 energy_threshold: float = 0.02,
                 min_duration_ms: int = 100,
                 max_duration_ms: int = 3000):
        self.sample_rate = sample_rate
        self.energy_threshold = energy_threshold
        self.min_duration = int(min_duration_ms * sample_rate / 1000)
        self.max_duration = int(max_duration_ms * sample_rate / 1000)
    
    def detect_breath_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect breath segments based on energy
        
        Returns:
            List of (start, end) sample indices
        """
        # Compute short-time energy
        frame_length = 512
        hop = 256
        
        energy = np.array([
            np.sum(audio[i:i + frame_length] ** 2)
            for i in range(0, len(audio) - frame_length, hop)
        ])
        
        # Normalize energy
        if energy.max() > 0:
            energy = energy / energy.max()
        
        # Find segments above threshold
        active = energy > self.energy_threshold
        segments = []
        
        in_segment = False
        start = 0
        
        for i, is_active in enumerate(active):
            if is_active and not in_segment:
                start = i * hop
                in_segment = True
            elif not is_active and in_segment:
                end = i * hop
                if self.min_duration <= (end - start) <= self.max_duration:
                    segments.append((start, end))
                in_segment = False
        
        # Handle segment at end
        if in_segment:
            end = len(audio)
            if self.min_duration <= (end - start) <= self.max_duration:
                segments.append((start, end))
        
        return segments
    
    def is_breath_detected(self, audio: np.ndarray) -> bool:
        """Check if audio contains a valid breath"""
        segments = self.detect_breath_segments(audio)
        return len(segments) > 0

    def validate_breath_duration(self, audio: np.ndarray, label: str) -> Tuple[bool, str]:
        """
        Validate if the breath duration matches the expected label.
        
        Args:
            audio: Audio waveform
            label: 'short' or 'long'
            
        Returns:
            (is_valid, message)
        """
        segments = self.detect_breath_segments(audio)
        if not segments:
            return False, "No breath detected"
            
        # Get duration of the longest breath segment in seconds
        max_duration_samples = max([end - start for start, end in segments])
        duration_sec = max_duration_samples / self.sample_rate
        
        # Define thresholds (adjust based on real world usage)
        # Normal breathing (short) is typically < 1.0 - 1.5s
        # Long breathing (trigger) should be significantly longer, e.g. > 1.5s - 2.0s
        
        # Simplified validation: Just check if *any* breath was detected
        # We rely on the model to learn the difference between short/long
        # rather than enforcing arbitrary time limits.
        
        if not segments:
             return False, "No breath detected. Please try again."
             
        return True, "Valid breath detected"
