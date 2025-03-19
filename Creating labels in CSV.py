
'''
Creating Labels CSV File

Format of the CSV file:

   file     ,  label
video1.mp4  ,  REAL
video2.mp4  ,  FAKE

'''
import os
import pandas as pd

def get_file_names(folder_path, label):
    file_names = os.listdir(folder_path)
    labeled_files = [(file_name, label) for file_name in file_names]
    return labeled_files

fake_folder_path = "path/to/fake/videos"
real_folder_path = "path/to/real/videos"

fake_files = get_file_names(fake_folder_path, "FAKE")
real_files = get_file_names(real_folder_path, "REAL")
all_files = fake_files + real_files

df = pd.DataFrame(all_files, columns=["File Names", "Label"])
df.to_csv("label.csv", index=False)
