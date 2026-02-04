import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def visualize_audio(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found {file_path}")
        return

    # Load audio
    y, sr = librosa.load(file_path, sr=16000)
    
    # Calculate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
    
    # 1. Waveform
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='blue')
    ax[0].set(title=f'Waveform: {os.path.basename(file_path)}')
    ax[0].label_outer()

    # 2. Spectrogram
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax[1])
    ax[1].set(title='Mel-Spectrogram')
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')

    # Save
    output_path = file_path.replace('.wav', '_viz.png')
    plt.savefig(output_path)
    plt.close()
    
    print(f"Generated visualization: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_audio.py <file_path>")
    else:
        visualize_audio(sys.argv[1])
