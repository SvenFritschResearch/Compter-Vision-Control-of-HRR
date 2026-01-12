import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import os
close_fig = False  # Global flag for closing the figure for plot_func_3D_video


def plot_func_3D(fk, fk1=None, fk2=None, desired_pose=None):
    #fk is a list of transformation matices starting at the base


    x_list = [mat[0,3] for mat in fk]
    y_list = [mat[1,3] for mat in fk]
    z_list = [mat[2,3] for mat in fk]

   
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    #ax.scatter3D(x_list, y_list, z_list, color = "green", marker=".")
    #ax.scatter3D(0,0,0, color = "red", marker="o")

    
    ax.plot(x_list, y_list, z_list, color = "orange", marker=".", label = "without null-space projection")
    
    if fk1 is not None:
        x_list_1 = [mat[0,3] for mat in fk1]
        y_list_1 = [mat[1,3] for mat in fk1]
        z_list_1 = [mat[2,3] for mat in fk1]
        ax.plot(x_list_1, y_list_1, z_list_1, color = "purple", marker=".", label = "with null-space projection")

    if fk2 is not None:
        x_list_2 = [mat[0,3] for mat in fk2]
        y_list_2 = [mat[1,3] for mat in fk2]
        z_list_2 = [mat[2,3] for mat in fk2]
        ax.plot(x_list_2, y_list_2, z_list_2, color = "teal", marker=".", label = "start configuration")

    
    

    # add coordinate frame to visualize orientation of end-effector 
    R = fk[-1][:3, :3]
    translation = fk[-1][:3, 3]
    scale = 70
    ax.quiver(*translation, *R[:, 0], color='blue', label='X-Axis EE', length=scale, linewidth=3)
    ax.quiver(*translation, *R[:, 1], color='green', label='Y-Axis EE', length=scale, linewidth=3)
    ax.quiver(*translation, *R[:, 2], color='black', label='Z-Axis EE', length=scale, linewidth=3)

    if desired_pose is not None:
        ax.plot([0, 0], [0, desired_pose[1]], [0, 0], color='grey', linestyle='--')
        ax.plot([0, desired_pose[0]], [desired_pose[1], desired_pose[1]], [0,0], color='grey', linestyle='--')
        ax.plot([desired_pose[0], desired_pose[0]], [desired_pose[1], desired_pose[1]], [0, desired_pose[2]], color='grey', linestyle='--')
            
        # add coordinate frame to visualize orientation of desired pose
        position = desired_pose[:3]
        orientation_euler = desired_pose[3:]
        rotation_matrix = Rotation.from_euler('xyz', orientation_euler, degrees=False).as_matrix()
        scale = 120  # Adjust the scale factor as needed
        ax.quiver(*position, *rotation_matrix[:, 0], color='lightblue', label='X-Axis des', length=scale)
        ax.quiver(*position, *rotation_matrix[:, 1], color='lightgreen', label='Y-Axis des', length=scale)
        ax.quiver(*position, *rotation_matrix[:, 2], color='grey', label='Z-Axis des', length=scale)

    
    # if desired_pose is not None :
    #     ax.scatter3D(desired_pose[0], desired_pose[1], desired_pose[2], color="red", marker="x", label="target pose", s=100)

    ax.axes.set_xlim3d(left=0, right=700) 
    ax.axes.set_ylim3d(bottom=-350, top=350) 
    ax.axes.set_zlim3d(bottom=-350, top=350)
    

    # Label the axes (optional)
    ax.set_xlabel('X [mm]', fontsize=14)
    ax.set_ylabel('Y [mm]', fontsize=14)
    ax.set_zlabel('Z [mm]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Null-space projection", fontsize=16)
    ax.view_init(elev=122, azim=-90)
    
    ax.legend()
    plt.show()



def on_key(event):
    """Closes the figure when ESC is pressed."""
    global close_fig
    if event.key == 'escape':  # Check if the ESC key is pressed
        plt.close('all')  # Close all figures
        close_fig = True  # Set flag to stop further plotting


def plot_func_3D_video(fk, fig=None, combined_image=None, update=False, save_video=False, update_counter=False):
    global close_fig
    if close_fig:
        return None

    if fig is None:
        plt.ion()  # Enable interactive mode
        fig = plt.figure(figsize=(20, 10))  # Width=20, Height=10
        # Use GridSpec: 2 rows, 3 columns, equal height for plots and image
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])  # Equal height for both rows

        # Three 3D plots in the top row
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')  # Top row, first column
        ax2 = fig.add_subplot(gs[0, 1], projection='3d')  # Top row, second column
        ax3 = fig.add_subplot(gs[0, 2], projection='3d')  # Top row, third column
        
        # Image plot in the bottom row, spanning all columns
        if combined_image is not None:
            ax_img = fig.add_subplot(gs[1, :])  # Bottom row, full width
        else:
            ax_img = None

        # Connect key press event
        fig.canvas.mpl_connect('key_press_event', on_key)
    else:
        plt.figure(fig.number)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
        ax3 = fig.axes[2]
        ax1.clear()
        ax2.clear()
        ax3.clear()
        if combined_image is not None and len(fig.axes) > 3:
            ax_img = fig.axes[3]
            ax_img.clear()
        else:
            ax_img = None

    # Extract positions from transformation matrices
    x_list = [mat[0, 3] for mat in fk]
    y_list = [mat[1, 3] for mat in fk]
    z_list = [mat[2, 3] for mat in fk]

    # Plot the 3D trajectories
    ax1.plot(x_list, y_list, z_list, color="chocolate", marker=".", label="IKM Jacobian Inverse")
    ax2.plot(x_list, y_list, z_list, color="chocolate", marker=".", label="IKM Jacobian Inverse")
    ax3.plot(x_list, y_list, z_list, color="chocolate", marker=".", label="IKM Jacobian Inverse")

    # Set axis limits
    ax1.set_xlim3d(0, 700)
    ax1.set_ylim3d(-350, 350)
    ax1.set_zlim3d(-350, 350)
    
    ax2.set_xlim3d(0, 700)
    ax2.set_ylim3d(-350, 350)
    ax2.set_zlim3d(-350, 350)

    ax3.set_xlim3d(0, 700)
    ax3.set_ylim3d(-350, 350)
    ax3.set_zlim3d(-350, 350)

    # Add coordinate frame for end-effector orientation
    R = fk[-1][:3, :3]
    translation = fk[-1][:3, 3]
    scale = 70
    ax1.quiver(*translation, *R[:, 0], color='blue', label='X-Axis EE', length=scale, linewidth=3)
    ax1.quiver(*translation, *R[:, 1], color='green', label='Y-Axis EE', length=scale, linewidth=3)
    ax1.quiver(*translation, *R[:, 2], color='black', label='Z-Axis EE', length=scale, linewidth=3)

    ax2.quiver(*translation, *R[:, 0], color='blue', label='X-Axis EE', length=scale, linewidth=3)
    ax2.quiver(*translation, *R[:, 1], color='green', label='Y-Axis EE', length=scale, linewidth=3)
    ax2.quiver(*translation, *R[:, 2], color='black', label='Z-Axis EE', length=scale, linewidth=3)

    ax3.quiver(*translation, *R[:, 0], color='blue', label='X-Axis EE', length=scale, linewidth=3)
    ax3.quiver(*translation, *R[:, 1], color='green', label='Y-Axis EE', length=scale, linewidth=3)
    ax3.quiver(*translation, *R[:, 2], color='black', label='Z-Axis EE', length=scale, linewidth=3)

    # Label axes
    ax1.set_xlabel('X Axis [mm]')
    ax1.set_ylabel('Y Axis [mm]')
    ax1.set_zlabel('Z Axis [mm]')
    ax1.view_init(elev=40, azim=50, roll=135)
    ax1.set_box_aspect([1, 1, 1])

    ax2.set_xlabel('')
    ax2.set_ylabel('Y Axis [mm]')
    ax2.set_zlabel('Z Axis [mm]')
    ax2.view_init(elev=0, azim=180, roll=-90)
    ax2.set_xticks([])
    ax2.set_box_aspect([1, 1, 1])

    ax3.set_xlabel('X Axis [mm]')
    ax3.set_ylabel('Y Axis [mm]')
    ax3.set_zlabel('')
    ax3.view_init(elev=90, azim=90, roll=-180)
    ax3.set_zticks([])
    ax3.set_box_aspect([1, 1, 1])

    ax1.legend()
    ax2.legend()
    ax3.legend()

    # Add combined image below the plots if provided
    if combined_image is not None and ax_img is not None:
        # Convert BGR (cv2) to RGB (matplotlib)
        combined_image_rgb = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
        ax_img.imshow(combined_image_rgb)
        ax_img.axis('off')  # Hide axes for the image
        ax_img.set_title('Camera Feed')

    # Adjust layout to prevent overlap
    plt.tight_layout()


    if update:
        plt.draw()
        plt.pause(0.001)
        if save_video:
            image_directory = r"C:\Dateien\3_ProTUTech_WiMi\Paper 3\Python Code\images"
            file_name = f"plot_{update_counter}.png"
            full_file_path = os.path.join(image_directory, file_name)
            dpi = 200
            plt.savefig(full_file_path, dpi=dpi)
    else:
        plt.show(block=False)

    return fig



def plot_error(error_list, error_threshold, k):
    counter = range(0, len(error_list))
    plt.plot(counter, error_list, marker=".", label="Error")
    plt.plot(counter, np.full(len(error_list), error_threshold), color="red", label="Error Threshold")
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Error', fontsize=14)
    plt.title(f'k={k}', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12, loc='upper center')
    plt.show()



def plot_time_boxplot(time_list_euler_random, time_list_quat_random, time_list_euler_path, time_list_quat_path):
    

    average_euler = np.mean(time_list_euler_random)
    average_quaternion = np.mean(time_list_quat_random)
    average_euler_path = np.mean(time_list_euler_path)
    average_quaternion_path = np.mean(time_list_quat_path)
    

    print(f"Average time for euler: {average_euler:.3f} s")
    print(f"Average time for quat: {average_quaternion:.3f} s")
    print(f"Average time for euler path: {average_euler_path:.3f} s")
    print(f"Average time for quat path: {average_quaternion_path:.3f} s")


    # Create a box plot for the distribution of times
    plt.figure(figsize=(4, 6))  # Adjust the width and height of the plot
    box = plt.boxplot([time_list_euler_random, time_list_quat_random, time_list_euler_path, time_list_quat_path], 
                       patch_artist=True)
    colors = ['green', 'red', 'blue', 'orange']

    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    plt.title('Convergence time', fontsize = 16)
    plt.yticks(fontsize=14)
    plt.xticks([1, 2, 3, 4], ['Euler angles \n random', 'Quaternion \n random', "Euler angles \n trajectory", "Quaternion \n trajectory"], fontsize=14)
    plt.ylim(bottom=0)  # Set the y-axis to start from 0
    plt.ylabel('Time [s]', fontsize=14)

    plt.show()



def plot_helix_path_with_orientation(x_t, y_t, z_t, psi_x, psi_y, psi_z, fk_list):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(x_t, y_t, z_t, label='target trajectory', color='black', marker='.')
    
    for i in range(len(x_t)):
        R_z = np.array([[np.cos(psi_z[i]), -np.sin(psi_z[i]), 0],
                        [np.sin(psi_z[i]),  np.cos(psi_z[i]), 0],
                        [0                ,  0                 ,1]])
        R_y = np.array([[np.cos(psi_y[i]) , 0, np.sin(psi_y[i])],
                        [0                 , 1,                 0],
                        [-np.sin(psi_y[i]), 0, np.cos(psi_y[i])]])
        R_x = np.array([[1, 0                ,                  0],
                        [0, np.cos(psi_x[i]), -np.sin(psi_x[i])],
                        [0, np.sin(psi_x[i]), np.cos(psi_x[i])]])
        
        R = R_z @ R_y @ R_x
        direction = R[:, 0]  # Take the x-axis direction after rotation
        scale_a = 40
        ax.quiver(x_t[i], y_t[i], z_t[i], R[0, 0], R[1, 0], R[2, 0], color='r', length = scale_a, label='target X-axis' if i == 0 else "")
        ax.quiver(x_t[i], y_t[i], z_t[i], R[0, 1], R[1, 1], R[2, 1], color='g', length = scale_a, label='target Y-axis' if i == 0 else "")
        ax.quiver(x_t[i], y_t[i], z_t[i], R[0, 2], R[1, 2], R[2, 2], color='b', length = scale_a, label='target Z-axis' if i == 0 else "")
        

    
    for m in range(len(fk_list)):

        x_list = [mat[0,3] for mat in fk_list[m]]
        y_list = [mat[1,3] for mat in fk_list[m]]
        z_list = [mat[2,3] for mat in fk_list[m]]
        ax.plot(x_list, y_list, z_list, color = "chocolate", marker=".")
        

        R = fk_list[m][-1][:3, :3]
        translation = fk_list[m][-1][:3, 3]
        scale = 50
        ax.quiver(*translation, *R[:, 0], color='r', length=scale, linewidth=5)
        ax.quiver(*translation, *R[:, 1], color='g', length=scale, linewidth=5)
        ax.quiver(*translation, *R[:, 2], color='b', length=scale, linewidth=5)


    ax.set_xlabel('X [mm]', fontsize=14)
    ax.set_ylabel('Y [mm]', fontsize=14)
    ax.set_zlabel('Z [mm]', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title("Helix trajectory", fontsize=16)
    ax.legend(fontsize=14, loc='lower left')
    plt.show()


