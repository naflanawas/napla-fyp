#!/usr/bin/env python3
"""
Debug script to visualize audio processing pipeline
Usage: python debug_audio.py <path_to_wav_file>
"""
import sys
import requests
import json

def debug_audio(audio_file_path, user_id="default_user", server_url="http://localhost:8000"):
    """Send audio to debug endpoint and print results"""
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': ('audio.wav', f, 'audio/wav')}
        response = requests.post(f"{server_url}/debug/{user_id}", files=files)
    
    if response.status_code == 200:
        data = response.json()
        
        print("\n" + "="*60)
        print("🎤 AUDIO PROCESSING PIPELINE DEBUG")
        print("="*60)
        
        print("\n📊 AUDIO INFO:")
        print(f"  Duration: {data['audio_info']['duration_sec']:.2f}s")
        print(f"  Samples: {data['audio_info']['num_samples']}")
        
        print("\n🎵 MEL-SPECTROGRAM (3-Channel: Spec + Delta + Delta²):")
        print(f"  Shape: {data['mel_spectrogram']['shape']}")
        print(f"  Frames: {data['mel_spectrogram']['num_frames']}")
        print(f"  Mel bins: {data['mel_spectrogram']['num_mels']}")
        
        print("\n🧠 EMBEDDING VECTOR:")
        print(f"  Dimension: {data['embedding']['dimension']}")
        print(f"  First 10 values: {[f'{v:.4f}' for v in data['embedding']['first_10_values']]}")
        print(f"  Mean: {data['embedding']['mean']:.4f}")
        print(f"  Std: {data['embedding']['std']:.4f}")
        
        print("\n🎯 PREDICTION:")
        print(f"  Intent: {data['prediction']['intent']}")
        print(f"  Confidence: {data['prediction']['confidence']:.4f}")
        print(f"  Distance: {data['prediction']['distance']:.4f}")
        print(f"  Is Confident: {data['prediction']['is_confident']}")
        
        print("\n📏 DISTANCES TO ALL PROTOTYPES:")
        for intent, distance in sorted(data['all_distances'].items(), key=lambda x: x[1]):
            marker = "✅" if intent == data['prediction']['intent'] else "  "
            print(f"  {marker} {intent:20s}: {distance:8.4f}")
        
        print("\n📋 CALIBRATED INTENTS:")
        print(f"  {', '.join(data['calibrated_intents'])}")
        
        print("\n" + "="*60)
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_audio.py <path_to_wav_file>")
        sys.exit(1)
    
    debug_audio(sys.argv[1])
