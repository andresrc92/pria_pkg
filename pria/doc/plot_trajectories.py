import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

def plot_3d_trajectories(file_paths):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a list of colors for different files
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # Iterate over each file path provided
    for idx, file_path in enumerate(file_paths):
        # Load the .npz file
        npzfile = np.load(file_path)
        
        # Access the first array in the .npz file
        filename = npzfile.files[0]
        trajectory_data = npzfile[filename]
        
        # Ensure the trajectory_data has the shape (N, 3), where N is the number of points and 3 corresponds to x, y, z
        if trajectory_data.shape[1] != 3:
            raise ValueError(f"The file '{file_path}' does not contain (x, y, z) data.")
        
        # Extract x, y, z values
        x_values = trajectory_data[:, 0]
        y_values = trajectory_data[:, 1]
        z_values = trajectory_data[:, 2]
        
        # Choose a color from the list, cycle if there are more files than colors
        color = colors[idx % len(colors)]
        
        # Plot the trajectory
        ax.plot(x_values, y_values, z_values, color=color, label=f"Trajectory {idx+1}: {file_path}")
    
    # Label the axes
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Trajectory Plot")
    
    # Show legend
    ax.legend()
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <trajectory1.npz> <trajectory2.npz> ...")
    else:
        file_paths = sys.argv[1:]  # Collect all file paths provided as arguments
        plot_3d_trajectories(file_paths)
