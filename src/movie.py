import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the directory where your CSV files are stored and where to save images
csv_directory = 'output'
image_directory = 'plots'
if not os.path.exists(image_directory):
    os.makedirs(image_directory)

# Read and plot each CSV file
for i in range(10000):
    # Load the matrix from a CSV file
    file_path = os.path.join(csv_directory, f'output_data_{i}.csv')
    matrix = pd.read_csv(file_path, header=0, index_col=0)  # Assuming no header in CSV

    # Create a heatmap from the matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation='nearest', cmap='hot')
    plt.colorbar()
    plt.title(f'Heatmap for Frame {i}')
    plt.savefig(os.path.join(image_directory, f'frame_{i:05d}.png'))
    plt.close()


os.system('ffmpeg -r 10 -i plots/frame_%05d.png -vcodec mpeg4 -y movie.mp4')
