import pandas as pd
import os


from modules import statsanalysis as stats
from modules import preprocessing as prep

df = pd.read_csv("data/IMS/1st_test/2003.11.23.02.46.56", sep="\t")

column_names = [f"Bearing 1 - Ch 1", f"Bearing 1 - Ch 2",
                f"Bearing 2 - Ch 1", f"Bearing 2 - Ch 2",
                f"Bearing 3 - Ch 1", f"Bearing 3 - Ch 2",
                f"Bearing 4 - Ch 1", f"Bearing 4 - Ch 2"]

features = prep.extract_features(df, chunk_size=10)


df = df.apply(pd.to_numeric, errors='coerce')

#Test AR


# df: ["t_mean", "fft_mean", "fft_max", "fft_spec_centroid", "fft_dominant_frequency", "fft_entropy"]
feature = df["fft_mean"]

print(feature)

aic, bic = stats.ARMA(feature, 1, 0)

print(aic, bic)


#Test ARMA

aic, bic = stats.ARMA(feature, 1, 1)

print(aic, bic)

