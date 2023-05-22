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

def load(filepath):

    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
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


def CreateSpectrograms(load_path,save_path, transformation = "MEL"):
    if transformation == 'MEL':
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
