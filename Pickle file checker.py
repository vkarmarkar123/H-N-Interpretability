import tkinter as tk
from tkinter import filedialog
import pickle

def load_pickle_file():
    # Create a root window, but don't display it
    root = tk.Tk()
    root.withdraw()

    # Open the file dialog and get the file path
    file_path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])

    # Check if a file was selected
    if file_path:
        # Open the file in binary read mode
        with open(file_path, 'rb') as file:
            # Load the contents from the file
            data = pickle.load(file)
        print(data)
    else:
        print("No file selected.")

# Call the function to load the pickle file
load_pickle_file()