import numpy as np
import pandas as pd
import os.path
import ast
import torchaudio
import torch

# Number of samples per 30s audio clip.
# TODO: fix dataset to be constant.
SAMPLING_RATE = 44100



class Genres:

    def __init__(self, tracks_df):
        self.df = tracks_df
    
    def GetGenres(self, path, genre = 'genre_top'):  #Returns two arrays, one with ID's the other with their genres
        id_list = []       
        genre_list = []                                
        for direc in list(path.iterdir()):
            if not direc.is_file():
                for file in (list(direc.iterdir())):
                    id_track = str(file)[-10:-4]
                    id_list.append(id_track)
                    genre_list.append(self.df.loc[self.df.track_id == int(id_track),genre].values[0])
        return np.asarray(id_list),np.asarray(genre_list)

def load(filepath): #loads CSV file from the specified filepath and performs different operations based on the filename

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
                                            #if it contains 'features' or 'echonest', it reads the CSV file with specific headers
    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename: #if it contains 'genres', it reads the CSV file with a basic index column
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename: #if it contains 'tracks', it reads the CSV file with multiple headers and performs additional data transformations before returning the loaded data
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def CreateSpectrograms(load_path,save_path, transformation = "MEL"): #  creates spectrograms from audio files and saves them as Torch tensors 
    if transformation == 'MEL':                                         # using either MelSpectrogram or Spectrogram transformation (default is MelSpectrogram)

        transform = torchaudio.transforms.MelSpectrogram(SAMPLING_RATE,n_fft=2048,hop_length=512)
    else:
        transform = torchaudio.transforms.Spectrogram(SAMPLING_RATE,n_fft=2048,hop_length=512)
    for direc in list(load_path.iterdir()):
        if not direc.is_file():
            for file in (list(direc.iterdir())):
                id_track = str(file)[-10:-4]
                try:
                    waveform, sample_rate = torchaudio.load(file)
                    if waveform.shape[0] > 1:
                        waveform = (waveform[0] + waveform[1])/2
                    spec = transform(waveform)
                    torch.save(spec, save_path+"/"+id_track+".pt")
                except:
                    pass

def ChargeDataset(path,id_list,genre_list):
    images = []
    labels = []
    for spec in list(path.iterdir()):
        id_track = str(spec)[13:-3]
        labels.append(genre_list[np.argwhere(id_list == id_track)][0][0])
        images.append(torch.load(spec))
    return np.asarray(images),np.asarray(labels)

def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)

class CustomSpectrogramDataset(Dataset):
    def __init__(self, spectrogram,genre, transform=None):
        self.x = spectrogram
        self.target = genre
        self.transform = transform

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.target[idx]
        if (self.transform!=None):
            image = self.transform(image)
        return image, label

def FixSpectrogramSize(spectrograms,genres,size): # reescale or cut spectograms so that they are all the same size
    spectograms_list = []
    genres_list = []
    for i,spec in enumerate(spectograms):
        if spec.shape == (128,2812):
            spectograms_list.append(spec[0:128,0:size])
            genres_list.append(genres[i])
            
        elif spec.shape == (1,128,size):
            spectograms_list.append(spec.reshape(128,size))
            genres_list.append(genres[i])

        elif spec.shape == (1,128,2585):
            spec = spec.reshape(128,2585)
            spectograms_list.append(spec[0:128,0:size])
            genres_list.append(genres[i])

        elif spec.shape == (128, 2585):
            spectograms_list.append(spec[0:128,0:size])
            genres_list.append(genres[i])

        elif spec.shape == (128, size):
            spectograms_list.append(spec)
            genres_list.append(genres[i])
    
    return spectograms_list, genres_list
    
