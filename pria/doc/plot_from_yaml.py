import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
import sys

def max_distance_from_point(point, points_vector):
    """
    Calculate the maximum distance between a given point and a vector of points.

    Args:
    - point: A list or array of shape (3,) representing the x, y, z coordinates of the point.
    - points_vector: A 2D array of shape (N, 3) representing N points with x, y, z coordinates.

    Returns:
    - max_distance: The maximum distance between the given point and the points in points_vector.
    """
    # Ensure both point and points_vector are numpy arrays
    point = np.array(point)
    points_vector = np.array(points_vector)
    
    # Compute the Euclidean distance between the point and all points in points_vector
    distances = np.linalg.norm(points_vector - point, axis=1)
    
    # Return the maximum distance
    max_distance = np.max(distances)
    
    return max_distance

def plot_trajectories_from_yaml(file_path, trajectory_num=None, just_plane=False):
    # Load the YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    end_points_cono = []
    end_points_plano = []
    
    # Define color variations for the trajectories
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    
    # If a specific trajectory number is provided, only plot that trajectory
    trajectories_to_plot = [trajectory_num] if trajectory_num is not None else list(data.keys())
    
    for trajectory_number in trajectories_to_plot:
        trajectory = data[trajectory_number]
        if not trajectory:
            print(f"Trajectory {trajectory_number} not found in the file.")
            continue
        
        # Plot 'cono' if it exists and has valid shape
        if 'cono' in trajectory and not just_plane:
            cono = np.array(trajectory['cono'])
            if cono.size != 0 and cono.shape[1] == 3:
                ax.plot(cono[:, 0], cono[:, 1], cono[:, 2], color=colors[int(trajectory_number) % len(colors)], label=f'Trajectory {trajectory_number} - Cono')
                # ax.plot(cono[-5:, 0], cono[-5:, 1], cono[-5:, 2], color=colors[int(trajectory_number) % len(colors)], label=f'Trajectory {trajectory_number} - Cono')
                # Add special points at the start and end of the 'cono' trajectory
                ax.scatter(cono[0, 0], cono[0, 1], cono[0, 2], color='black', s=50, marker='o', label=f'Trajectory {trajectory_number} Start (Cono)')
                ax.text(cono[0, 0], cono[0, 1], cono[0, 2], str(trajectory_number), zdir=(1,1,0),fontsize=15)
                # ax.scatter(cono[-5, 0], cono[-5, 1], cono[-5, 2], color='black', s=50, marker='o', label=f'Trajectory {trajectory_number} Start (Cono)')
                # ax.text(cono[-5, 0], cono[-5, 1], cono[-5, 2], str(trajectory_number), zdir=(1,1,0),fontsize=15)

                ax.scatter(cono[-1, 0], cono[-1, 1], cono[-1, 2], color='orange', s=50, marker='X', label=f'Trajectory {trajectory_number} End (Cono)')
                end_points_cono.append(cono[-1])
            else:
                print(f"Skipping 'cono' for trajectory {trajectory_number} due to invalid shape or empty data.")
        
        # Plot 'plano' if it exists and has valid shape
        if 'plano' in trajectory:
            plano = np.array(trajectory['plano'])
            if plano.size != 0 and plano.shape[1] == 3:
                ax.plot(plano[:, 0], plano[:, 1], plano[:, 2], color=colors[int(trajectory_number) % len(colors)], linestyle='--', label=f'Trajectory {trajectory_number} - Plano')
                # Add special points at the start and end of the 'plano' trajectory
                ax.scatter(plano[0, 0], plano[0, 1], plano[0, 2], color='black', s=50, marker='o')
                ax.scatter(plano[-1, 0], plano[-1, 1], plano[-1, 2], color='orange', s=50, marker='X')
                end_points_plano.append(plano[-1])
            else:
                print(f"Skipping 'plano' for trajectory {trajectory_number} due to invalid shape or empty data.")

    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)
    ax2d.axis("equal")

    fig_alturas = plt.figure()
    ax_alturas = fig_alturas.add_subplot(111)

    plane_center = np.mean(end_points_plano, axis=0)
    print("Punto central plano", plane_center)
    print("Radio plano 3d", max_distance_from_point(plane_center, end_points_plano))
    puntos_finales_plano = np.array(end_points_plano)[:, 2]

    puntos_finales_filtrados = np.delete(puntos_finales_plano,[5,8])
    print("Media de alturas plano ", np.mean(np.array(end_points_plano)[:, 2]))
    print("Media de alturas filtradas ", np.mean(puntos_finales_filtrados))

    ax_alturas.scatter(np.array(end_points_plano)[:,1], np.array(end_points_plano)[:,2], color='orange', s=50, marker='X', label=f'Alturas finales')
    for i in range(np.array(end_points_plano).shape[0]):
        ax_alturas.annotate(str(i), (np.array(end_points_plano)[i,1], np.array(end_points_plano)[i,2]), textcoords='offset points', xytext=(-5,5))

    end_points_plano -= plane_center
    ax2d.scatter(np.array(end_points_plano)[:, 0], np.array(end_points_plano)[:, 1], color='orange', s=50, marker='X', label=f'Final trajectorias plana')

    for i in range(np.array(end_points_plano).shape[0]):
        ax2d.annotate(str(i), (np.array(end_points_plano)[i,0], np.array(end_points_plano)[i,1]), textcoords='offset points', xytext=(-5,5))
    
    ax2d.scatter(0, 0, color='orange', s=100, marker='*', label=f'Final trajectorias cono')
    plane_radius_plane = max_distance_from_point([0,0], end_points_plano[:,:2])

    circle_plane = plt.Circle((0,0),plane_radius_plane, color='orange', fill=False)
    ax2d.add_patch(circle_plane)


    print("Radio plano 2d", plane_radius_plane)
    print("Desviación plano", np.std(end_points_plano, axis=0))


    if not just_plane:

        ax_alturas.scatter(np.array(end_points_cono)[:,1], np.array(end_points_cono)[:,2], color='black', s=50, marker='o', label=f'Alturas finales')
        for i in range(np.array(end_points_cono).shape[0]):
            ax_alturas.annotate(str(i), (np.array(end_points_cono)[i,1], np.array(end_points_cono)[i,2]), textcoords='offset points', xytext=(-5,5))
        
        puntos_finales_cono = np.array(end_points_cono)[:, 2]
        puntos_finales_filtrados = np.delete(puntos_finales_cono,[5,8])
        print("Media de alturas cono ", np.mean(np.array(end_points_cono)[:, 2]))
        print("Media de alturas filtradas ", np.mean(puntos_finales_filtrados))
        end_points_cono -= plane_center
        cone_center = np.mean(end_points_cono, axis=0)
        ax2d.scatter(np.array(end_points_cono)[:, 0], np.array(end_points_cono)[:, 1], color='black', s=50, marker='o', label=f'Final trajectorias cono')
        
        for i in range(np.array(end_points_cono).shape[0]):
            ax2d.annotate(str(i), (np.array(end_points_cono)[i,0], np.array(end_points_cono)[i,1]), textcoords='offset points', xytext=(-5,5))
        
        ax2d.scatter(cone_center[0], cone_center[1], color='black', s=100, marker='*', label=f'Centro trajectorias cono')
    
        plane_radius_cone = max_distance_from_point(cone_center[:2], end_points_cono[:,:2])
        circle_cone = plt.Circle(cone_center[:2],plane_radius_cone, color='black', fill=False)
        ax2d.add_patch(circle_cone)
    
        plane_radius_cone = max_distance_from_point(cone_center[:2], end_points_cono[:,:2])



        print("Punto central cono", cone_center)
        print("Radio cono", max_distance_from_point(cone_center, end_points_cono))
        print("Radio cono 2d", plane_radius_cone)
        print("Desviación cono", np.std(end_points_cono, axis=0))


    
    # Label the axes
    ax2d.set_xlabel("X axis (m)", fontsize=16)
    ax2d.set_ylabel("Y axis (m)", fontsize=16)

    ax_alturas.set_xlabel("X axis (m)", fontsize=16)
    ax_alturas.set_ylabel("Z axis (m)", fontsize=16)

    ax.set_xlabel("X axis (m)")
    ax.set_ylabel("Y axis (m)")
    ax.set_zlabel("Z axis (m)")
    # ax.set_title("3D Trajectories from YAML")
    
    # Show legend
    # ax.legend()
    
    # Display the plot

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py <trajectories.yaml> [trajectory_num]")
    else:
        yaml_file = sys.argv[1]
        trajectory_num = sys.argv[2] if len(sys.argv) > 2 else None
        
        # If a trajectory number is provided, convert it to int; else, keep it None
        trajectory_num = int(trajectory_num) if trajectory_num else None
        plot_trajectories_from_yaml(yaml_file, trajectory_num)
