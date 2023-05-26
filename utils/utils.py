import numpy as np
import pandas as pd
import os.path
import torch
from pathlib import Path,PureWindowsPath,PurePosixPath
import matplotlib.pyplot as plt
import librosa
from torch.utils.data.dataloader import DataLoader, Dataset
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Number of samples per 30s audio clip.
SAMPLING_RATE = 44100


    
def GetGenres(path,dict_genre,tracks ,genre_att = 'genre_top'):  #Returns two arrays, one with ID's the other with their genres
    id_list = []       
    genre_list = []                                
    for direc in list(path.iterdir()):
        if direc.is_file():
            id_track = str(direc)[-10:-4]
            id_list.append(id_track)
            genre_list.append(dict_genre[tracks.loc[tracks.track_id == int(id_track),genre_att].values[0]])
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
    #if transformation == 'MEL':                                         # using either MelSpectrogram or Spectrogram transformation (default is MelSpectrogram)

     #   transform = torchaudio.transforms.MelSpectrogram(SAMPLING_RATE,n_fft=2048,hop_length=512)
    #else:
     #   transform = torchaudio.transforms.Spectrogram(SAMPLING_RATE,n_fft=2048,hop_length=512)
    for file in list(load_path.iterdir()):
        id_track = str(file)[18:-3]
        
        try:
            waveform, sample_rate = librosa.load(file)
            spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft = 2048,hop_length=512 )
            if waveform.shape[0] > 1:
                    waveform = (waveform[0] + waveform[1])/2
            torch.save(spec, str(save_path)+"/"+id_track+".pt")
            
        except:
            pass

def ChargeDataset(path,id_list,genre_list):
    images = []
    labels = []
    for i,spec in enumerate(list(path.iterdir())):
        id_track = str(spec)[18:-3]
        labels.append(genre_list[np.argwhere(id_list == id_track)][0][0])
        
        spec = torch.load(spec)
        spec = np.asarray(librosa.power_to_db(spec))
        images.append(spec)
        
    return images,labels


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
   # plt.show(block=False)

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
    for i,spec in enumerate(spectrograms):
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

def FixSizeSpectrogram(spectrograms,genres,shapes):
    spectograms_list = []
    genres_list = []
    height = shapes[0]



    for i,spec in enumerate(spectrograms):
        if spec.shape != (shapes[0],shapes[1]):
            spectograms_list.append(spec[0:shapes[0],0:shapes[1]])
            genres_list.append(genres[i])
            
        else:
            spectograms_list.append(spec)
            genres_list.append(genres[i])
    return spectograms_list, genres_list

def LoadFixCSV():
    tracks = pd.read_csv("./data/tracks.csv", low_memory=False)
    genres = pd.read_csv("./data/genres.csv")
    tracks.columns=tracks.iloc[0] 
    tracks.columns.values[0] = "track_id"
    tracks.drop([0,1],inplace=True)
    tracks.track_id = tracks.track_id.astype(int)

    return tracks,genres

def CreateTrainTestLoaders(spectrograms_list, genres_list, train_size, train_kwargs, test_kwargs):
    #Faltaria afegir split de test i train 
    train_mean = np.mean(spectrograms_list)/255. #Mean of all images
    train_std = np.std(spectrograms_list)/255. 
    
    #transform = transforms.Compose([
        #transforms.Normalize((train_mean,), (train_std,))
        #])

    X_train, X_val, y_train, y_val = train_test_split(spectrograms_list, genres_list, train_size=train_size, stratify=genres_list)

    train_ds = CustomSpectrogramDataset(X_train, y_train)
    test_ds = CustomSpectrogramDataset(X_val, y_val)

    train_dataloader = torch.utils.data.DataLoader(train_ds, **train_kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_ds, **test_kwargs)


    return train_dataloader, test_dataloader, y_val #i tambe el test_dataloader


def LoadDataPipeline():
    

    tracks, genres = LoadFixCSV()
    print("Tracks and Genres loaded")
    genre_dict = {'Electronic':0,'Experimental':1,'Folk':2,'Hip-Hop':3,
             'Instrumental':4, 'International':5, 'Pop':6, 'Rock':7}

    path = Path("./data/audio")
    id_list, genre_list = GetGenres(path,genre_dict,tracks)
    save_path = Path("./data/Spectrograms")
    if len(list(save_path.iterdir())) != 7994:
        CreateSpectrograms(path,save_path)
    print("Spectrograms created")
    spectrograms, genres = ChargeDataset(save_path,id_list,genre_list)
    print("Spectrograms loaded")

    shape = []
    for i in spectrograms:
        shape.append(i.shape)
    
    shapes = np.unique(shape)


    spectrograms_list, genres_list = FixSizeSpectrogram(spectrograms,genres,shapes)
    print("Size fixed for Spectrograms")


    return spectrograms_list, genres_list

def visualize_confusion_matrix(y_pred, y_real):
    #mostra la matriu de confusió
    cm = confusion_matrix(y_real, y_pred)
    plt.subplots(figsize=(10, 6))
    sns.heatmap(cm, annot = True, fmt = 'g')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
