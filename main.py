import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit


import tensorflow as tf
from tensorflow import keras

from models.AnomalyDetector import AnomalyDetector


def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs 
    normal_cutoff = cutoff / nyquist  
    b, a = butter(order, normal_cutoff, btype='low', analog=False) 
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data) 

def feature_extraction(b=3, path="data/IMS/1st_test/", save_to_csv=False, redo_features=False):
    
    bearings = {
        "bearing_1": [0, 1],
        "bearing_2": [2, 3],
        "bearing_3": [4, 5],
        "bearing_4": [6, 7]
    }

    bearing = bearings[f"bearing_{b}"]

    N = 20000    # Chunk size
    T = 1 / 20e3 
    fs = 20e3 # Sampling frequency
    

    features = ["t_mean", "fft_mean", "fft_max", "fft_spec_centroid", "fft_dominant_frequency", "fft_entropy"]


    feature_file = os.path.join(path, f"features_{b}.csv")

    if not os.path.isfile(feature_file) or redo_features == True:

        filenames = []

        for (root, _, files) in os.walk(path):
            for file in files:
                if file.startswith('200'):
                    filenames.append(os.path.join(root,file)) 


        filenames.sort()
        i = 0
        
        for file in filenames:

            print(f"parsing file: {i}, out of {len(filenames)}")
            i = i + 1


            df = pd.read_csv(file, sep="\t",usecols=bearing, dtype= {bearing[0] : "float32", bearing[1] : "float32" })

            cutoff = 5000

            
            N = len(df)

            df.columns = ["X", "Y"]
            df["Mag"] = np.sqrt(df["X"]**2 + df["Y"]**2)

            
            
            df["gauss"] = gaussian_filter1d(df["Mag"], sigma=5)

            df["filtered"] = butter_lowpass_filter(df["gauss"], cutoff, fs)

            yf = fft(df["filtered"])
            # xf = fftfreq(N, T)[:N//2]
            f, Pxx = welch(df["filtered"].values, fs=fs, nperseg=512)

            feature = [np.mean(df["filtered"]), 
                        np.mean(np.abs(yf)[5:]),
                        np.max(np.abs(yf)[5:]), 
                        np.sum(f * Pxx) / np.sum(Pxx),
                        f[np.argmax(Pxx)],
                        entropy(Pxx / np.sum(Pxx))
                    ]

            df_features = pd.DataFrame([feature], columns=features)

            if save_to_csv == True:
                df_features.to_csv(feature_file, mode="a", header=False, index=False)
    
    df_features.columns = ["t_mean", "fft_mean", "fft_max", "fft_spec_centroid", "fft_dominant_frequency", "fft_entropy"]

    return df_features

def preprocessing(df, save_to_csv=False, b=3, use_normalization=False, trim_start=True, trim_val=175, path="data/IMS/1st_test/"):
    
    if trim_start == True or trim_val != 175:
        df = df.iloc[trim_val:] 

    if use_normalization == True:
        df_processed = (df - df.min()) / (df.max() - df.min())

        if save_to_csv == True:
            normalized_file = os.path.join(path, f"features_normalized_{b}.csv")
            df_processed.to_csv(normalized_file, mode="a", index=False)

    else:

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)
        df_processed = pd.DataFrame(X_scaled, columns=df.columns)
        if save_to_csv == True:
            scaled_file = os.path.join(path, f"features_scaled_{b}.csv")
            df_processed.to_csv(scaled_file, index=False)

    return df_processed

def test_train_data(df):

    transition_start = 1200
    transition_end = 1800


    X_train = df.iloc[:transition_start]
    X_val = df.iloc[transition_start:transition_end]
    X_test = df.iloc[transition_end:]

    # tscv = TimeSeriesSplit(n_splits=5)

    # for train_index, test_index in tscv.split(df):
    #    X_train, X_test = df.iloc[train_index], df.iloc[test_index]

    return X_train, X_test, X_val


def create_model(train_data, test_data):

    #Use autoencoder, use autoencoder + LSTM


    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    # train_labels = train_labels.astype(bool)
    # test_labels = test_labels.astype(bool)

    normal_train_data = train_data # [train_labels]
    normal_test_data = test_data # [test_labels]

    # anomalous_train_data = train_data[~train_labels]
    # anomalous_test_data = test_data[~test_labels]

    autoencoder = AnomalyDetector()

    autoencoder.compile(optimizer='adam', loss='mae')

    history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=20,
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
    

    return history


def main():
    print("hello")




if __name__ == "__main__":
    main()