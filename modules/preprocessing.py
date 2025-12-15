
import numpy as np
import pandas as pd
import os
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import entropy
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler


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
    
    # features = ["Crest Factor", "RMS", "Impulse Factor", "Margin Factor", "Shape Factor", "Peak-Peak Value"]

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

def std_scaler(df, b, save_to_csv=False, trim_start=True, trim_val=175, path="data/IMS/1st_test/"):
    if trim_start == True or trim_val != 175:
        df = df.iloc[trim_val:] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    df_processed = pd.DataFrame(X_scaled, columns=df.columns)
    if save_to_csv == True:
        scaled_file = os.path.join(path, f"features_scaled_{b}.csv")
        df_processed.to_csv(scaled_file, index=False)

    return df_processed

def norm_scaler(df, b, save_to_csv=False, trim_start=True, trim_val=175, path="data/IMS/1st_test/"):
    if trim_start == True or trim_val != 175:
        df = df.iloc[trim_val:] 
        
    df_processed = (df - df.min()) / (df.max() - df.min())

    if save_to_csv == True:
        normalized_file = os.path.join(path, f"features_normalized_{b}.csv")
        df_processed.to_csv(normalized_file, mode="a", index=False)
    
    return df_processed


def preprocessing(df, save_to_csv=False, b=3, use_normalization=True, trim_start=True, trim_val=175, path="data/IMS/1st_test/"):
    
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



def extract_features(df, chunk_size=10):
    
    df_temp = df.head(0)
    features = pd.DataFrame()

    df_temp['subchunk'] = df.index // chunk_size
    

    # Compute magnitude per bearing
    for i in range(len(df_temp.columns)):
        mag = np.sqrt(df_temp.iloc[:,2*i]**2 + df_temp.iloc[:,2*i + 1]**2)
        mag_mean = mag.groupby(df_temp['subchunk']).mean().reset_index(drop=True)
        features[f"Magnitude {i}"] = mag_mean




    return features