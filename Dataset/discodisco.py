import base64
import requests
import json
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import re  

# Step 1: Set up your Spotify Developer Account and get your Client ID and Client Secret
client_id = 'c037e76b11504ab596ecbfcb11a6ae73'
client_secret = '95b360ec038e4c91b6c0bd49f1063f67'

# Step 2: Get an access token
def get_access_token(client_id, client_secret):
    auth_url = 'https://accounts.spotify.com/api/token'
    auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        'Authorization': f'Basic {auth_header}',
    }
    data = {
        'grant_type': 'client_credentials',
    }
    response = requests.post(auth_url, headers=headers, data=data)
    response.raise_for_status()  # Raise an error for bad status codes
    return response.json().get('access_token')

# Step 3: Search for tracks in a specific genre
def search_tracks_by_genre(access_token, genre, limit=50, offset=0):
    search_url = f'https://api.spotify.com/v1/search?q=genre:{genre}&type=track&limit={limit}&offset={offset}'
    headers = {
        'Authorization': f'Bearer {access_token}',
    }
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    return response.json()

# Function to sanitize file names
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename)

# Step 4: Extract Track Information and Download Previews
def download_previews(tracks, download_dir='previews'):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    for item in tracks['items']:
        track = item['track']
        preview_url = track['preview_url']
        if preview_url:
            track_name = sanitize_filename(track['name'])
            artist_name = sanitize_filename(track['artists'][0]['name'])
            file_name = f"{artist_name} - {track_name}.mp3"
            file_path = os.path.join(download_dir, file_name)
            response = requests.get(preview_url)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded: {file_path}")

# Step 5: Transform Previews into GTZAN-like Melspectrograms without x and y axis labels and ensure they are 432x288 pixels
def create_melspectrograms(download_dir='previews', output_dir='melspectrograms-hiphop', duration=30, sample_rate=22050):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
    
    for file_name in os.listdir(download_dir):
        if file_name.endswith('.mp3'):
            file_path = os.path.join(download_dir, file_name)
            print(f"Processing file: {file_path}")
            y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
            
            # Ensure the audio is 30 seconds long by padding or trimming
            if len(y) < duration * sr:
                y = np.pad(y, (0, duration * sr - len(y)), mode='constant')
            else:
                y = y[:duration * sr]
            
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(2.88, 2.88), dpi=100)  # Set figure size to 288x288 pixels
            librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)  # Remove x and y axis labels
            plt.axis('off')  # Turn off the axis
            plt.tight_layout(pad=0)  # Remove padding
            
            output_file_path = os.path.join(output_dir, file_name.replace('.mp3', '.png'))
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved melspectrogram: {output_file_path}")

def rename_files(output_dir):
    # List all files in the output directory
    files = os.listdir(output_dir)
    
    # Sort files to ensure consistent renaming
    files.sort()
    
    # Rename each file to "1.png", "2.png", "3.png", and so on
    for i, file_name in enumerate(files, start=1):
        old_file_path = os.path.join(output_dir, file_name)
        new_file_name = f"{i}.png"
        new_file_path = os.path.join(output_dir, new_file_name)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

# Specify the output directory
output_dir = r"C:\Users\Steve Fuentes\melspectrograms-jazz"

# Call the function to rename files
rename_files(output_dir) 

# Main function to automate the process
def main():
    access_token = get_access_token(client_id, client_secret)
    
    genre = 'hip-hop'
    limit = 50
    total_tracks = 1000
    all_tracks = []
    
    for offset in range(0, total_tracks, limit):
        result = search_tracks_by_genre(access_token, genre, limit=limit, offset=offset)
        tracks = result.get('tracks', {}).get('items', [])
        if tracks:
            all_tracks.extend(tracks)
    
    download_previews({'items': [{'track': track} for track in all_tracks]})
    create_melspectrograms()

if __name__ == '__main__':
    main()