from __future__ import unicode_literals
import yt_dlp as youtube_dl
from typing import List

def download_wav(links: List[str], save_to: str, use_genre: str = '') -> None:
    """
    This function downloads the given list of youtube links, keeping audio only, in .wav format, and saves it to the specified directory.
    You can also specify the genre to save.
    Please pass save path as string.
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

download_wav(['https://www.youtube.com/watch?v=dQw4w9WgXcQ'], save_to=r'Dataset/downloads', use_genre='disco')