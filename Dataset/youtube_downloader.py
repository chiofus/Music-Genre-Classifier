from __future__ import unicode_literals
import yt_dlp as youtube_dl
from typing import List
import os
from pydub import AudioSegment
import librosa
from librosa import get_duration, load
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def download_wav(links: List[str], save_to: str, use_genre: str = '') -> List[str]:
    """
    This function downloads the given list of youtube links, keeping audio only, in .wav format, and saves it to the specified directory.
    You can also specify the genre to save.
    Please pass save path as string.

    Returns a list containing all the unique identifiers for the video downloads.
    """

    if use_genre != '':
        use_genre += '/' #adding extra slash when not empty, so it creates a dir correctly (otherwise it would be missing a slash)

    ydl_opts = {
    'format': 'bestaudio/best',
    'ffmpeg_location': './ffmpeg/ffmpeg-2024-10-31-git-87068b9600-full_build/bin/ffmpeg.exe',
    'outtmpl': f'./{save_to}/{use_genre}%(title)s-%(id)s.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192'
    }],
    'postprocessor_args': [
        '-ar', '16000'
    ],
    'prefer_ffmpeg': True,
    'keepvideo': False
}

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download(links)

    
    identifiers: List[str] = list()
    for link in links:
        identifiers.append(link.split("watch?v=")[-1])

    return identifiers

def wav_splitter(file: Path, save_to: Path = Path(os.getcwd()), split_length_ms: int = 30000) -> Path:
    """
    Splits the given .wav file into as many clips possible for the given split length.

    Please specify the save_to path for custom save directory.

    Returns directory for the downloaded segments.
    """

    filename: str = file.as_posix().split('/')[-1].split('.')[0] #gets filename of the given song
    savepath: Path = save_to/filename #directory to save all the segments to

    song_segment: AudioSegment = AudioSegment.from_wav(file) #creating segment object that can be split
    max_length: float = get_duration(path=file) * 1000 #gets total duration of song, in miliseconds
    
    starting_at: float = 0.0
    curr_split: int = 1

    #Creating save path, in case it does not exist
    if not os.path.exists(save_to):
        os.mkdir(save_to) #creating genre directory

    if not os.path.exists(savepath): 
        os.mkdir(savepath) #creating specific song directory

    while starting_at != max_length:
        curr_segment: AudioSegment = song_segment[starting_at: starting_at+split_length_ms] #gets split for current segment
        curr_segment.export(savepath/f'{filename}_{curr_split}.wav', format='wav') #saves current segment to files
        starting_at = min(starting_at+split_length_ms, max_length) #setting next length to be min of total song length of past iteration + split length
        curr_split += 1

    return savepath

def create_melspectrograms(files_dir: Path, output_dir: Path, sample_rate: int = 22050, duration_secs: float = 30.0) -> None:
    """
    Creates spectrograms for all .wav files it can find in the given files_dir.

    Please specify dir to save spectrograms to in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
    
    for file_name in os.listdir(files_dir): #listing all files in given directory
        if file_name.endswith('.wav'): #checking that file is a .wav audio file
            file_path = files_dir/file_name #getting full path of current file
            if get_duration(path=file_path) != duration_secs:
                continue #skipping files that are not 30 secs in length

            print(f"Processing file: '{file_path.as_posix()}'")
            y, sr = load(file_path, sr=sample_rate)
            
            # # Ensure the audio is 30 seconds long by padding or trimming
            # if len(y) < duration * sr:
            #     y = np.pad(y, (0, duration * sr - len(y)), mode='constant')
            # else:
            #     y = y[:duration * sr]
            
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(3, 3), dpi=100)  # Set figure size to 288x288 pixels
            librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)  # Remove x and y axis labels
            plt.axis('off')  # Turn off the axis
            plt.tight_layout(pad=0)  # Remove padding
            
            output_file_path = (output_dir/file_name).as_posix().replace(".wav", ".png") #creating new save path for spectrogram with .png extension
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            print(f"Saved melspectrogram: {output_file_path}")

def get_spectrograms_for_links(links: List[str], genre: str,
                               wav_savepath: str = r"Dataset\downloads", 
                               splits_savepath: str = r"Dataset\split_wavs_from_downloads",
                               spectrograms_savepath: str = r"Dataset\youtube_spectrograms") -> None:
    """
    For all the given links, downloads them and turns them into as many 30 second clips as possible.

    These clips are then turned into spectrograms.
    """

    #Downloading audio for given links as .wav
    identifiers = download_wav(links, save_to= wav_savepath, use_genre=genre) #getting identifiers for newly downloaded links
    print(identifiers)

    #Splitting
    for donwloaded_filename in os.listdir(Path(wav_savepath)/genre): #iterating through all song dowloads
        for identifier in identifiers:
            if identifier in donwloaded_filename: #finding a match for identifier from all the downloads; found newly downloaded song
                print(f"Now processing for: '{donwloaded_filename}'")
                #splitting file
                savepath = wav_splitter(file= Path(wav_savepath)/genre/donwloaded_filename, save_to=Path(splits_savepath)/genre)

                #creating spectrograms
                create_melspectrograms(files_dir=savepath, output_dir=Path(spectrograms_savepath)/genre)



get_spectrograms_for_links(links=["https://www.youtube.com/watch?v=GPMcwzESL7I",
                                  "https://www.youtube.com/watch?v=mvCgSqPZ4EM"], genre='country')
# create_melspectrograms(files_dir=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\split_wavs_from_downloads\disco\Rick Astley - Never Gonna Give You Up (Official Music Video)-dQw4w9WgXcQ"),
#                        output_dir=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset")/"youtube_spectrograms"/"disco")
# download_wav(['https://www.youtube.com/watch?v=dQw4w9WgXcQ'], save_to=r'Dataset/downloads', use_genre='disco')
# wav_splitter(file=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\downloads\disco\Rick Astley - Never Gonna Give You Up (Official Music Video)-dQw4w9WgXcQ.wav"),
            #  save_to= Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\split_wavs_from_downloads\disco"))