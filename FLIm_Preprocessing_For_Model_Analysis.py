import os
import pandas as pd
from Fast_BEX import optimized_bex
import xgboost as xgb
import matplotlib.pyplot as plt
import shap
import pickle
import tkinter as tk
from tkinter import filedialog

df = pd.read_csv("./CSVs and Output Files/HN_P2C_20231207.csv")
cwd = os.getcwd()
# filename = 'Anatomic_DT_LG'
# output_dir = os.path.join(cwd, filename)
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
df = df[df['ScanContext'] == 'In Vivo']

# %% SNR and lifetime filtering
index = df[((df['snr_ch1'] < 30) & (df['Instrument'] == 'V4')) |
           ((df['snr_ch2'] < 30) & (df['Instrument'] == 'V4')) |
           ((df['snr_ch3'] < 30) & (df['Instrument'] == 'V4'))].index
df.drop(index, inplace=True)
index = df[((df['snr_ch1'] < 50) & (df['Instrument'] == 'FLImBrush')) |
           ((df['snr_ch2'] < 50) & (df['Instrument'] == 'FLImBrush')) |
           ((df['snr_ch3'] < 50) & (df['Instrument'] == 'FLImBrush'))].index
df.drop(index, inplace=True)

# R_data((R_data.lifet_avg_ch1 > 10 | R_data.lifet_avg_ch1 < 2),:) = [];
index = df[((df['lifet_avg_ch1'] > 16) | (df['lifet_avg_ch1'] < 1)) |
           ((df['lifet_avg_ch2'] > 16) | (df['lifet_avg_ch2'] < 1)) |
           ((df['lifet_avg_ch3'] > 16) | (df['lifet_avg_ch3'] < 1))].index
df.drop(index, inplace=True)

# %% For our current analysis we remove all V4 data and no channel 4 data
index = df[df['Instrument'] == 'FLImBrush'].index
df.drop(index, inplace=True)

# df = df[df['Anatomy'] =='Palatine Tonsil']
# df = df[df['Anatomy'] =='Base of Tongue'] | df[df['Anatomy'] =='Lingual Tonsil'] | df[df['Anatomy'] =='Palatine Tonsil']
# df = df[df['Anatomy'] == 'Superior Tongue']

# df = df[df['Anatomy'].isin(['Base of Tongue', 'Lingual Tonsil', 'Palatine Tonsil'])]


# Extract columns based on specific patterns
pattern = r'(Case|Run|snr_|lifet_avg_|spec_int_|int_ratio_|Laguerre_coeffs_|Phasor_GH|Phasor_SH|WLI_)'
selected_columns = df.columns[df.columns.str.contains(pattern)]

# Add the "Label" column, which doesn't fit the patterns
selected_columns = selected_columns.insert(len(selected_columns), 'Label')

subset_df = df[selected_columns]
subset_df = subset_df.dropna()

subset_df = subset_df[subset_df['Label'] != 0]
# Replace values in the 'Label' column
subset_df['Label'] = subset_df['Label'].replace([1, 2, 18, 20, 25, 27, 29, 36, 37], 0)
subset_df['Label'] = subset_df['Label'].replace([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 24, 28, 30, 33], 1)

# Remove rows where 'Label' is not 0 or 1
subset_df = subset_df[subset_df['Label'].isin([0, 1])]

bex_df = optimized_bex(subset_df)
# bex_df = bex(subset_df)
f = 1

# Bex(subset_df, l_k=48, f_k1=6, f_k2=47)
# %% For our current analysis we remove all V4 data and no channel 4 data
from scipy.spatial.distance import pdist

pattern = r'(Case|Run|snr_|lifet_avg_|spec_int_|int_ratio_|Laguerre_coeffs_|WLI_)'
selected_columns = df.columns[df.columns.str.contains(pattern)]
selected_columns = selected_columns.insert(len(selected_columns), 'Label')
TrainTest_Data = bex_df[selected_columns]


# Here you can filter the data based on case number

# ----------------------------SHAP Analysis----------------------------


# Open .pkl files
def load_pickle_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])

    if file_path:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        root.destroy()
        return data
    else:
        print("No file selected.")
        root.destroy()
        return None


data = load_pickle_file()
xgb_model = data['model']
dataset = subset_df

# SHAP Explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(dataset)
shap.summary_plot(shap_values, dataset)
