import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2

def convert_npy_to_png(root_dir, output_dir, target_size=(432, 288)):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Walk through the directory
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.npy'):
                npy_path = os.path.join(subdir, file)
                # Load the .npy file
                array = np.load(npy_path)
                # Resize the array to the target size
                resized_array = cv2.resize(array, target_size)
                # Create the output path
                relative_path = os.path.relpath(subdir, root_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)
                png_path = os.path.join(output_subdir, os.path.splitext(file)[0] + '.png')
                # Save the array as an image with a magenta color map
                plt.figure(figsize=(target_size[0] / 100, target_size[1] / 100))
                librosa.display.specshow(resized_array, sr=22050, x_axis='time', y_axis='mel', cmap='magma')
                plt.axis('off')
                plt.savefig(png_path, bbox_inches='tight', pad_inches=0)
                plt.close()
                print(f'Saved {png_path}')

# Set the root directory containing the .npy files
root_directory = r'D:\APS360-DATA\\rough\\melspecs-00-21'
# Set the output directory for the .png files
output_directory = r'D:\APS360-DATA\\rough\\melspecs-img-00-09-final-colour-lib-3'
convert_npy_to_png(root_directory, output_directory)


