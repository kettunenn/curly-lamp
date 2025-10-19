import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.stats import entropy


import examples.features as f

def preprocessing(x):
    
    bearings = {
        "bearing_1": [0, 1],
        "bearing_2": [2, 3],
        "bearing_3": [4, 5],
        "bearing_4": [6, 7]
    }

    bearing = bearings[f"bearing{x}"]

    N_tot = len(df)
    N = 20000    # Chunk size
    T = 1 / 20e3 # Sampling frequency

    

    features = ["t_mean", "fft_mean", "fft_max", "fft_spec_centroid", "fft_dominant_frequency", "fft_entropy"]

    
    path = "../data/IMS/1st_test/"
    feature_file = os.path.join(path, f"features_{x}.csv")

    if not os.path.isfile(feature_file):

        filenames = []

        for (root, _, files) in os.walk(path):
            for file in files:
                if file.startswith('200'):
                    filenames.append(os.path.join(root,file)) 

        filenames.sort()

        for file in filenames:
            df = pd.read_csv(file, sep="\t",usecols=bearing, dtype= {bearing[0] : "float32", bearing[1] : "float32" })
            df.columns = ["X", "Y"]

            df["Mag"] = np.sqrt(df["X"]**2 + df["Y"]**2)


            yf = fft(df["Mag"])
            # xf = fftfreq(N, T)[:N//2]

            f, Pxx = welch(x, fs=T, nperseg=1024)
            

            feature = [np.mean(df["Mag"]), 
                       np.max(yf), 
                       np.mean(yf), 
                       np.sum(f * Pxx) / np.sum(Pxx),
                       f[np.argmax(Pxx)],
                       entropy(Pxx / np.sum(Pxx))
                       ]

            
            df_features = pd.DataFrame(feature, columns=features)

            df_features.to_csv(feature_file, mode="a", index=False)

    
def main():
    print("hello")




if __name__ == "__main__":
    main()