#!/usr/bin/env python3
"""
Batch debug script to analyze multiple audio files
Usage: python batch_debug.py /path/to/recordings/folder
"""
import sys
import requests
import json
from pathlib import Path

def debug_audio(audio_file_path, server_url="http://localhost:8000"):
    """Send audio to debug endpoint and return results"""
    
    with open(audio_file_path, 'rb') as f:
        files = {'file': ('audio.wav', f, 'audio/wav')}
        response = requests.post(f"{server_url}/debug/default_user", files=files)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def analyze_batch(folder_path):
    """Analyze all WAV files in a folder"""
    
    folder = Path(folder_path)
    wav_files = sorted(folder.glob("*.wav"))
    
    if not wav_files:
        print(f"No WAV files found in {folder_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"📊 BATCH ANALYSIS: {len(wav_files)} files")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, wav_file in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] Processing: {wav_file.name}")
        
        data = debug_audio(wav_file)
        
        if data:
            results.append({
                'filename': wav_file.name,
                'duration': data['audio_info']['duration_sec'],
                'intent': data['prediction']['intent'],
                'confidence': data['prediction']['confidence'],
                'distance': data['prediction']['distance'],
                'all_distances': data['all_distances']
            })
        else:
            print(f"  ❌ Error processing {wav_file.name}")
        
        print()
    
    # Summary table
    print(f"\n{'='*80}")
    print("📋 SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'File':<30} {'Duration':>8} {'Intent':>8} {'Conf':>6} {'Dist':>8}")
    print(f"{'-'*80}")
    
    for r in results:
        filename_short = r['filename'][:28]
        print(f"{filename_short:<30} {r['duration']:>7.2f}s {r['intent']:>8} {r['confidence']:>5.0%} {r['distance']:>8.2f}")
    
    print(f"{'-'*80}")
    
    # Distance analysis
    print(f"\n{'='*80}")
    print("📏 DISTANCE ANALYSIS")
    print(f"{'='*80}")
    
    avg_distance = sum(r['distance'] for r in results) / len(results)
    min_distance = min(r['distance'] for r in results)
    max_distance = max(r['distance'] for r in results)
    
    print(f"Average Distance: {avg_distance:.2f}")
    print(f"Min Distance:     {min_distance:.2f} (Best match)")
    print(f"Max Distance:     {max_distance:.2f} (Worst match)")
    
    if avg_distance < 10:
        print("✅ Excellent consistency! Your breaths are very similar.")
    elif avg_distance < 20:
        print("⚠️  Moderate variation. Consider re-calibrating for more consistent breaths.")
    else:
        print("❌ High variation. Your breath patterns are inconsistent.")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_debug.py /path/to/recordings/folder")
        sys.exit(1)
    
    analyze_batch(sys.argv[1])
