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
import torch.nn as nn
from torch import Tensor
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Subset
from tools import save_features
import torch
import shutil
from tools import genre_classifier
from torchvision import models
from typing import Dict
import sys

#Pytorch, plotting
import torch
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import Tensor
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.utils.data import Subset

#General/other
from typing import Set, Tuple, List
import numpy as np
import random, time
from pathlib import Path
import os, math

#Loading functions

from tools import augment_training, plot_training_curve, get_model_name, get_correct, get_model_path, evaluate, seed_worker, load_randomized_loaders
from tools import save_features, train_classifier, genre_classifier, plot_confusion_matrix

classification_dict = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}
classification_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

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
    'ffmpeg_location': r'W:\school\aps360\APS360-Music-Genre-Classifier\ffmpeg\ffmpeg-2024-10-31-git-87068b9600-full_build\bin\ffmpeg.exe', #please change to your custom ffmpeg location
    'outtmpl': f'./{save_to}/{use_genre}%(id)s.%(ext)s',
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

    When save one is true, only saves first or second segment of song. Saves first when song is too short to make two full segments,
    otherwise only saves second segment.
    """

    filename: str = file.as_posix().split('/')[-1].split('.')[0] #gets filename of the given song
    savepath: Path = save_to/filename #directory to save all the segments to

    song_segment: AudioSegment = AudioSegment.from_wav(file) #creating segment object that can be split
    max_length: float = get_duration(path=file) * 1000 #gets total duration of song, in miliseconds
    n_segments: float = max_length/split_length_ms #gets how many segmens will be created
    
    starting_at: float = 0.0
    curr_split: int = 1

    #Creating save path, in case it does not exist
    if not os.path.exists(save_to):
        os.mkdir(save_to) #creating genre directory

    if not os.path.exists(savepath): 
        os.mkdir(savepath) #creating specific song directory

    done: bool = False
    while starting_at != max_length:
        curr_segment: AudioSegment = song_segment[starting_at: starting_at+split_length_ms] #gets split for current segment

        curr_segment.export(savepath/f'{filename}_{curr_split}.wav', format='wav') #saves current segment to files
        
        starting_at = min(starting_at+split_length_ms, max_length) #setting next length to be min of total song length of past iteration + split length
        curr_split += 1


        if done:
            break #exiting when save one condition was satisfied

    return savepath

def scale_arr(arr: np.ndarray, min: float = 0.0, max: float = 1.0) -> np.ndarray:
    """
    Scales all values in given arr in the range of min and max values provided.
    """
    arr_min: float = arr.min()
    arr_max: float = arr.max()

    #Standardizing arr with min max normalization (all values will now be in the range 0 - 1)
    x_std: np.ndarray = (arr - arr_min) / (arr_max - arr_min)

    #Scaling with given max, min
    x_scaled: np.ndarray = x_std * (max - min) + min

    return x_scaled

def create_melspectrograms(files_dir: Path, output_dir: Path, sample_rate: int = 22050, duration_secs: float = 30.0,
                           dimensions: Tuple[float, float] = (2.88, 2.88)) -> None:
    """
    Creates spectrograms for all .wav files it can find in the given files_dir.

    Please specify dir to save spectrograms to in output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")

    n_fft = 2048
    n_mels = 128
    hop_length = 512
    
    for file_name in os.listdir(files_dir): #listing all files in given directory
        if file_name.endswith('.wav'): #checking that file is a .wav audio file
            file_path = files_dir/file_name #getting full path of current file
            if get_duration(path=file_path) != duration_secs:
                continue #skipping files that are not 30 secs in length

            print(f"Processing file: '{file_path.as_posix()}'")
            y, sr = librosa.load(file_path, sr=sample_rate)
            
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=dimensions, dpi=100)  # Set figure size to given dimensions
            librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)  # Remove x and y axis labels
            plt.axis('off')  # Turn off the axis
            plt.tight_layout(pad=0)  # Remove padding
            
            output_file_path = (output_dir/file_name).as_posix().replace(".wav", ".png") #creating new save path for spectrogram with .png extension
            plt.savefig(output_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            # plt.figure(figsize=dimensions, dpi=100)  # Set figure size to 288x288 pixels
            # librosa.display.specshow(S_dB, sr=sr, x_axis=None, y_axis=None)  # Remove x and y axis labels
            # plt.axis('off')  # Turn off the axis
            # plt.tight_layout(pad=0)  # Remove padding
            
            print(f"Saved melspectrogram: {output_file_path}")

def get_spectrograms_from_links(links: List[str], genre: str,
                                dimensions: Tuple[float, float],
                               wav_savepath: str = r"Dataset\downloads", 
                               splits_savepath: str = r"Dataset\split_wavs_from_downloads",
                               spectrograms_savepath: str = r"Dataset\youtube_spectrograms") -> None:
    """
    For all the given links, downloads them and turns them into as many 30 second clips as possible.

    These clips are then turned into spectrograms.

    When only save one is passed as true, only the second spectrogram is saved, as the first one may not have enough information for a relevant
    prediciton. When the song is too short to have two full segments, the first one is saved.
    """

    #Downloading audio for given links as .wav
    identifiers = download_wav(links, save_to= wav_savepath, use_genre=genre) #getting identifiers for newly downloaded links

    #Splitting
    for donwloaded_filename in os.listdir(Path(wav_savepath)/genre): #iterating through all song dowloads
        for identifier in identifiers:
            if identifier in donwloaded_filename: #finding a match for identifier from all the downloads; found newly downloaded song
                print(f": '{donwloaded_filename}'")
                #splitting file
                savepath = wav_splitter(file= Path(wav_savepath)/genre/donwloaded_filename, save_to=Path(splits_savepath)/genre)

                #creating spectrograms
                create_melspectrograms(files_dir=savepath, output_dir=Path(spectrograms_savepath)/genre, dimensions=dimensions)

def inferable_tensor_from_spectrogram(spectrogram_path: Path, encoder: nn.Module = None, complete: bool = True) -> Tensor:
    """
    Turns the given spectrogram image path into an inferable tensor.

    Note that the spectrogram path must point to a directory holding a single folder with the single image to turn into a tensor.
    If an encoder is given, returns its embedding after running through encoder.
    """

    #Creating dummy loader for given images

    #Defining transforms to apply to images
    trns: transforms.Compose = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    ) #Transforms images to tensors, then applies (img - 0.5)/0.5 to all channels to obtain data in range [-1, 1]

    data_folder: ImageFolder = ImageFolder(root=spectrogram_path.as_posix(), transform=trns) #creating dummy loader

    image_loader = DataLoader(data_folder, batch_size=1, shuffle=False)

    #Processing through encoder, when provided
    if not complete:
        if encoder != None:
            features = save_features(image_loader, encoder)

            #Returning first features from first batch
            return features[0][0] #accessing features directly

        #Else, return next iteration from loader
        image, label = next(iter(image_loader))
        return image #accessing image tensor directly from loader
    
    #Processing for multiple spectrograms
    complete_data: Tensor = Tensor()
    if encoder != None:
        features = save_features(image_loader, encoder)

        for feature, label in features:
            complete_data = torch.cat((complete_data, feature))

    else:
        for data, label in image_loader:
            complete_data = torch.cat(complete_data, data)
    
    return complete_data

def clone_spectrogram(genre: str, reference: str, clone_to: Path, spectrogram_nbr: int = 2,
                      clone_from: Path = Path(r"Dataset\youtube_spectrograms"),
                      complete: bool = True) -> None:
    """
    Clones the given spectrogram, at the given spectrogram number. Finds it with the given reference, at the given genre.

    Provide path to clone it to.

    When complete is true, infers on all spectrograms created from song, and final answer is the most repeated genre.
    """

    #Finding file to clone
    if not complete:
        while spectrogram_nbr != 0: #iterating until all spectrogram options are exhausted
            for filename in os.listdir(clone_from/genre): #looking in given directory
                if f"{reference}_{spectrogram_nbr}" in filename: #looking for a perfect segment and reference match
                    shutil.copy(src=clone_from/genre/filename, dst=clone_to/genre/filename)
                    return None
        
        print(f"No match found for '{reference}'")

    else:
        for filename in os.listdir(clone_from/genre): #looking in given directory
            if reference in filename: #looking for a perfect segment and reference match
                shutil.copy(src=clone_from/genre/filename, dst=clone_to/genre/filename)

    return None

def infer_genre_given_link(classifier: nn.Module, encoder: nn.Module, link: List[str],
                           dimensions: Tuple[float, float],
                           save_spectrogram_at: Path = Path(r"Dataset\youtube_spectrograms_inference"),
                           complete: bool = True) -> str:
    """
    Infers genre from given youtube link, using the given model.

    When complete is true, infers on all spectrograms created from song, and final answer is the most repeated genre.
    """

    if len(link) != 1: #does not support prediction for mutiple songs; simply used for a one song demonstration
        print("Prediction for more than one song at the same time not supported.")
        return 'invalid'

    #Preparing input for model
    get_spectrograms_from_links(links=link, genre='inference', dimensions = dimensions) #downloading and saving spectrogram
    
    reference: str = link[0].split("watch?v=")[-1] #unique link reference
    shutil.rmtree(save_spectrogram_at/"inference") #clearing inference directory before pasting isoalted spectrogram into it
    os.mkdir(save_spectrogram_at/"inference") #remaking inference directory
    clone_spectrogram(genre='inference', reference=reference, clone_to=save_spectrogram_at, complete = complete) #saving spectrograms to infer from
    
    data = inferable_tensor_from_spectrogram(spectrogram_path=save_spectrogram_at, encoder=encoder, complete= complete) #extracts Tensor to use for classifier
    if torch.cuda.is_available: #sending to gpu
        data = data.cuda()

    with torch.no_grad():
        results: Tensor = classifier(data)

    #Normalizing
    softmax = nn.Softmax()
    results = softmax(results)
    print(f"Results: {results}")

    #Reading prediction
    pred: str = ''
    if not complete:
        prediction_index: int = results.max(1, keepdim=True).indices.squeeze(dim=1).item() #reading single class prediction
        pred = classification_list[prediction_index]
    
    else:
        predictions = results.max(1, keepdim=True).indices.squeeze(dim=1)

        total_preds: Dict[str, int] = {
            'blues': 0,
            'classical': 0,
            'country': 0,
            'disco': 0,
            'hiphop': 0,
            'jazz': 0,
            'metal': 0,
            'pop': 0,
            'reggae': 0,
            'rock': 0
        } #keep count of how many times each unique genre was predicted

        for curr_pred in predictions:
            pred_index = curr_pred.item()
            total_preds[classification_list[pred_index]] += 1

        #Finding max
        max: int = 0
        print(total_preds)
        for key, val in total_preds.items():
            if val > max:
                pred = key
                max = val

        #Appending ties to show mixed genre
        for key, val in total_preds.items():
            if val == max and key != pred:
                pred += f"-{key}"

    return  pred #returns genre matching prediction

def infer(link: List[str], dimensions: Tuple[float, float] = (2.88, 2.88), complete: bool = True) -> str:
    #Loading encoders, loaders

    #Importing resnet, and modifying classifier.
    resnet152 = models.resnet152(pretrained=True)

    #Since resnet does not have 'features', we need to manually remove the last fc layer.
    modules = list(resnet152.children())[:-1] # delete the last fc layer.
    resnet152 = nn.Sequential(*modules)

    #Loading best model
    best_classifier = genre_classifier(dropout_prob=0.5, output_size= 2048 * 1 * 1, first_fc_dim=1024, second_fc_dim=512)

    if torch.cuda.is_available:
        best_classifier = best_classifier.cuda()

    model_path = r"W:\school\aps360\APS360-Music-Genre-Classifier\__Main__\resnet_transfer\training_data\model_genre_classifier_dr0.5_lr0.005494691666620257_epoch200_bs256\model_genre_classifier_dr0.5_lr0.005494691666620257_epoch69_bs256" #epoch 70 had the best validation accuracy
    state = torch.load(model_path)
    best_classifier.load_state_dict(state)

    #making inference
    return infer_genre_given_link(best_classifier, resnet152, link, dimensions = dimensions, complete = complete)

def infer_vgg16_yt(link: List[str], dimensions: Tuple[float, float] = (2.88, 2.88), complete: bool = True) -> str:
    #Loading encoders, loaders

    #Importing VGG16, and modifying classifier.
    vgg_16: nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)

    #Loading best model
    best_classifier = genre_classifier(dropout_prob=0.4, output_size= 9 * 9 * 512)

    if torch.cuda.is_available:
        best_classifier = best_classifier.cuda()

    model_path = r"W:\school\aps360\APS360-Music-Genre-Classifier\__Main__\vgg16_ytdata\training_data\model_genre_classifier_dr0.4_lr0.005494691666620257_epoch50_bs32\model_genre_classifier_dr0.4_lr0.005494691666620257_epoch13_bs32" #epoch 14 had the best validation accuracy
    state = torch.load(model_path)
    best_classifier.load_state_dict(state)

    #setting to evaluation mode
    # vgg_16.eval()
    # best_classifier.eval()

    #making inference
    return infer_genre_given_link(best_classifier, vgg_16, link, dimensions = dimensions, complete = complete)

if __name__ == "__main__":

    link = ["https://www.youtube.com/watch?v=i2FW1WJc0lg"] #provide link manually here

    try:
        link = [sys.argv[1]] #if link given in args, will override link above
    except:
        print("No link provided in args")

    prediction = infer_vgg16_yt(link)

    print(f"Prediction for '{link[0]}': {prediction}")
    # get_spectrograms_from_links(links=links, genre='rock', dimensions=(2.88, 2.88))
    # vgg_16: nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)

    # data = inferable_tensor_from_spectrogram(spectrogram_path=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\test"), encoder=vgg_16)
    # print(data, data.shape)
# vgg_16: nn.Module = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', pretrained=True)

# # data = inferable_tensor_from_spectrogram(spectrogram_path=Path(r"Dataset\youtube_spectrograms\country\test"), encoder=vgg_16)
# print(infer_genre_given_link())

# get_spectrograms_from_links(links=["https://www.youtube.com/watch?v=GPMcwzESL7I",
#                                   "https://www.youtube.com/watch?v=mvCgSqPZ4EM"], genre='country')
# create_melspectrograms(files_dir=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\split_wavs_from_downloads\disco\Rick Astley - Never Gonna Give You Up (Official Music Video)-dQw4w9WgXcQ"),
#                        output_dir=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset")/"youtube_spectrograms"/"disco")
# download_wav(['https://www.youtube.com/watch?v=dQw4w9WgXcQ'], save_to=r'Dataset/downloads', use_genre='disco')
# wav_splitter(file=Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\downloads\disco\Rick Astley - Never Gonna Give You Up (Official Music Video)-dQw4w9WgXcQ.wav"),
            #  save_to= Path(r"W:\school\aps360\APS360-Music-Genre-Classifier\Dataset\split_wavs_from_downloads\disco"))