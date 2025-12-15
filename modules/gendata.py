import pandas as pd
import os
import numpy as np


path = "../data/IMS/1st_test/"

savefig = True #Save plots

filenames = []

for (root, _, files) in os.walk(path):
    for file in files:
        if "2003" in file:
            filenames.append(os.path.join(root,file)) 

filenames.sort()




column_names = [f"Bearing 1 - Ch 1", f"Bearing 1 - Ch 2",
                f"Bearing 2 - Ch 1", f"Bearing 2 - Ch 2",
                f"Bearing 3 - Ch 1", f"Bearing 3 - Ch 2",
                f"Bearing 4 - Ch 1", f"Bearing 4 - Ch 2"]

df = pd.DataFrame(columns=column_names)


data_channels = ["Mean Bearing 1 - Ch 1", "Mean Bearing 1 - Ch 2",
                 "Mean Bearing 2 - Ch 1", "Mean Bearing 2 - Ch 2",
                 "Mean Bearing 3 - Ch 1", "Mean Bearing 3 - Ch 2",
                 "Mean Bearing 4 - Ch 1", "Mean Bearing 4 - Ch 2"]

Magnitude = ["Magnitude Bearing 1", 
             "Magnitude Bearing 2", 
             "Magnitude Bearing 3", 
             "Magnitude Bearing 4"]


df = pd.DataFrame()

chunk_size = 10

for file in filenames:
    df_temp = pd.read_csv(file, sep="\t", names=data_channels)
    
    df_temp['subchunk'] = df_temp.index // chunk_size
    df_mag = pd.DataFrame()

    # Compute magnitude per bearing
    for i in range(len(Magnitude)):
        mag = np.sqrt(df_temp[data_channels[2*i]]**2 + df_temp[data_channels[2*i + 1]]**2)
        mag_mean = mag.groupby(df_temp['subchunk']).mean().reset_index(drop=True)
        df_mag[Magnitude[i]] = mag_mean

    # Append this file's computed magnitudes
    df = pd.concat([df, df_mag], ignore_index=True)

# df.to_csv("mean values of data", mode="a", header=False, index=False) 


#Takes a 1-dim dataframe and applies function func 
# xyz are extra function arguments

# A chunk size of 10 equals 100ms and approximately 2000 datapoints
def file_to_df(path):
    return


def chunking(file, func, x,y,z, chunk_size=10):
    
    df_temp = pd.read_csv(file, sep="\t", names=data_channels)
    df_temp['chunk'] = df.temp.index // chunk_size

    return df
