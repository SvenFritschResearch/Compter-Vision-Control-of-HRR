import cv2
import numpy as np
import time
import pyzed.sl as sl
from kinematic_model_Paper3 import *
from plotting_numeric_Paper3 import *
time.sleep(2)  # Wait for Arduino to initialize


def print_q(q):
    print(f"q = [insert_rob: {q[0]:.2f} mm, alpha: {np.rad2deg(q[1]):.2f} deg, rob_len: {q[2]:.2f} mm, theta: {np.rad2deg(q[3]):.2f} deg, delta_l_niti: {q[4]:.2f} mm, rho_prox: {q[5]:.2f} deg, rho_distal: {q[6]:.2f} deg]")

class KalmanFilter3D:
    def __init__(self, process_variance=1e-3, measurement_variance=1e-1):
        self.x = np.array([[0.0], [0.0], [0.0]])  # Initial state: (X, Y, Z)
        self.P = np.eye(3)  # Covariance matrix
        self.Q = np.eye(3) * process_variance  # Process noise
        self.R = np.eye(3) * measurement_variance  # Measurement noise
        self.K = np.zeros((3, 3))  # Kalman Gain

    def update(self, measurement):
        """Performs Kalman filter update step."""
        measurement = np.array(measurement).reshape(3, 1)  # Convert to column vector
        self.P += self.Q  # Prediction step

        # Compute Kalman Gain
        self.K = self.P @ np.linalg.inv(self.P + self.R)

        # Update estimate
        self.x += self.K @ (measurement - self.x)
        self.P = (np.eye(3) - self.K) @ self.P  # Update uncertainty

        return self.x.flatten()  # Return as (X, Y, Z)
    

def get_X_e(q, design_paras):
    #fk_start = forward_kinematics_3D_redundant_quaternion(q, design_paras)
    fk_start = forward_kinematics_3D_redundant(q, design_paras)
    #plot_func_3D(fk_start)
    x_position = fk_start[-1][0,3]
    y_position = fk_start[-1][1,3]
    z_position = fk_start[-1][2,3]
    R = fk_start[-1][:3,:3]

    # return Z_Y_X euler angles
    psi_z = np.arctan2( R[1,0], R[0,0])
    psi_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
    psi_x = np.arctan2( R[2,1], R[2,2])
    X_e = np.array([x_position, y_position, z_position, psi_x, psi_y, psi_z]) # X_e = current pose parameterization of the end effector 
    
    return X_e


def send_converted_q_to_arduino(q_converted):

    # Convert list of positions to comma-separated string with newline
    command = ','.join(map(str, q_converted)) + '\n'
    ser.write(command.encode())  # Send the command
    print(f"Sent: {command.strip()}")
    
    # Wait for response from Arduino
    while True:
      response = ser.readline().decode().strip()
      if response:
        print(f"Arduino says: {response}")
        if "Enter next positions" in response or "Invalid input" in response:
          break



    
def retrieve_target_pose():
    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 60
    init_params.coordinate_units = sl.UNIT.MILLIMETER


    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        zed.close()
        exit(1)

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)


    #all focal lengths and c (optical axis) are in pixels
    if init_params.camera_resolution == sl.RESOLUTION.HD2K:
        focal_length_left_x = 1067.93 
        focal_length_left_y = 1068.33 
        focal_length_right_x = 1069.49
        focal_length_right_y = 1069.92
        cx_left = 1100.14
        cy_left = 622.994
        cx_right = 1098.58
        cy_right = 630.731
        
    if init_params.camera_resolution == sl.RESOLUTION.HD1080:
        focal_length_left_x = 1067.93  
        focal_length_left_y = 1068.33  
        focal_length_right_x = 1069.49 
        focal_length_right_y = 1069.92 
        cx_left = 956.14
        cy_left = 541.994
        cx_right = 954.58
        cy_right = 549.731

    if init_params.camera_resolution == sl.RESOLUTION.HD720:
        focal_length_left_x = 533.965  
        focal_length_left_y = 534.165  
        focal_length_right_x = 534.745 
        focal_length_right_y = 534.96  
        cx_left = 636.57
        cy_left = 359.497
        cx_right = 635.79
        cy_right = 363.3655

    if init_params.camera_resolution == sl.RESOLUTION.VGA:
        focal_length_left_x = 266.9825  
        focal_length_left_y = 267.0825  
        focal_length_right_x = 267.3725 
        focal_length_right_y = 267.48  
        cx_left = 333.785
        cy_left = 187.2485
        cx_right = 333.395
        cy_right = 189.18275

    baseline = 119.844  #mm

    image_left = sl.Mat()
    image_right = sl.Mat()
    start_time = time.time()
    only_do_once = True
    delta_target_pose_list = []
    motion_scaling_factor_position = 1
    motion_scaling_factor_orientation = 1
    countdown_seconds = 3
    it_counter = 0 #counter for the iterations of the while loop
    update_counter = 0 #counter for the updates of the robot pose
    raw_data_green_list = []
    filtered_data_green_list = []
    raw_data_blue_list = []
    filtered_data_blue_list = []

    # Design parameters
    len_distal_part = 130 #mm
    len_enddisk = 46 #mm
    len_wrist = 42 #mm
    dist_tendons= 9.7 #mm
    design_paras = [len_distal_part, len_enddisk, len_wrist, dist_tendons]

    q = [100, 0, 150, np.deg2rad(0), 0.01, 0, 0] #inital q, q=7x1 vector with q = [stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal]
    k = 0.5  # Gain
    error_threshold = 0.1

    fig = None #important for updating the plot, don't delete
    kf_3D_green = KalmanFilter3D()
    kf_3D_blue = KalmanFilter3D()



    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            it_counter += 1
            zed.retrieve_image(image_left, sl.VIEW.LEFT)
            zed.retrieve_image(image_right, sl.VIEW.RIGHT)

            left_array = image_left.get_data()
            right_array = image_right.get_data()


            image_left_hsv = cv2.cvtColor(left_array[:,:,:3], cv2.COLOR_BGR2HSV)
            image_right_hsv = cv2.cvtColor(right_array[:,:,:3], cv2.COLOR_BGR2HSV)


            # Define the range for the color red in HSV
            lower_blue = np.array([110, 120, 120])
            upper_blue = np.array([125, 255, 255])

            lower_green = np.array([50, 110, 90])
            upper_green = np.array([80, 255, 255])
            
            mask_filter_values_blue_left = cv2.inRange(image_left_hsv, lower_blue, upper_blue)
            mask_filter_values_green_left = cv2.inRange(image_left_hsv, lower_green, upper_green)
            mask_filter_values_blue_right = cv2.inRange(image_right_hsv, lower_blue, upper_blue)
            mask_filter_values_green_right = cv2.inRange(image_right_hsv, lower_green, upper_green)
            
            mask_filter_values_left = cv2.bitwise_or(mask_filter_values_blue_left, mask_filter_values_green_left)
            mask_filter_values_right = cv2.bitwise_or(mask_filter_values_blue_right, mask_filter_values_green_right)
            masked_image_left = cv2.cvtColor(cv2.bitwise_and(image_left_hsv, image_left_hsv, mask= mask_filter_values_left), cv2.COLOR_HSV2BGR)
            masked_image_right = cv2.cvtColor(cv2.bitwise_and(image_right_hsv, image_right_hsv, mask= mask_filter_values_right), cv2.COLOR_HSV2BGR)

            contours_for_blue_left, _ = cv2.findContours(mask_filter_values_blue_left, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE)
            contours_for_blue_right, _ = cv2.findContours(mask_filter_values_blue_right, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE)
            contours_for_green_left, _ = cv2.findContours(mask_filter_values_green_left, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE)
            contours_for_green_right, _ = cv2.findContours(mask_filter_values_green_right, cv2.RETR_TREE  , cv2.CHAIN_APPROX_SIMPLE)
            

            if contours_for_blue_left:
                largest_contour = max(contours_for_blue_left, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_blue_left_x = x + w // 2
                center_blue_left_y = y + h // 2
                cv2.drawMarker(masked_image_left, (center_blue_left_x, center_blue_left_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=4)
            
            if contours_for_blue_right:
                largest_contour = max(contours_for_blue_right, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_blue_right_x = x + w // 2
                center_blue_right_y = y + h // 2
                cv2.drawMarker(masked_image_right, (center_blue_right_x, center_blue_right_y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=4)

            if contours_for_green_left:
                largest_contour = max(contours_for_green_left, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_green_left_x = x + w // 2
                center_green_left_y = y + h // 2
                cv2.drawMarker(masked_image_left, (center_green_left_x, center_green_left_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=4)        

            if contours_for_green_right:
                largest_contour = max(contours_for_green_right, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                center_green_right_x = x + w // 2
                center_green_right_y = y + h // 2
                cv2.drawMarker(masked_image_right, (center_green_right_x, center_green_right_y), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=4)


            
            if contours_for_green_left and contours_for_green_right:
                disparity_green = center_green_left_x - center_green_right_x  

                Z_green_raw = (baseline * focal_length_left_x) / disparity_green
                X_green_raw = (center_green_left_x - cx_left) * Z_green_raw / focal_length_left_x
                Y_green_raw = (center_green_left_y - cy_left) * Z_green_raw / focal_length_left_y

                raw_data_green_list.append([X_green_raw, Y_green_raw, Z_green_raw])
                X_green, Y_green, Z_green = kf_3D_green.update([X_green_raw, Y_green_raw, Z_green_raw])
                filtered_data_green_list.append([X_green, Y_green, Z_green])
                
               

            if contours_for_blue_left and contours_for_blue_right:
                disparity_blue = center_blue_left_x - center_blue_right_x  

                Z_blue_raw = (baseline * focal_length_left_x) / disparity_blue
                X_blue_raw = (center_blue_left_x - cx_left) * Z_blue_raw / focal_length_left_x
                Y_blue_raw = (center_blue_left_y - cy_left) * Z_blue_raw / focal_length_left_y

                raw_data_blue_list.append([X_blue_raw, Y_blue_raw, Z_blue_raw])
                X_blue, Y_blue, Z_blue = kf_3D_blue.update([X_blue_raw, Y_blue_raw, Z_blue_raw])                
                filtered_data_blue_list.append([X_blue, Y_blue, Z_blue])


            # Print countdown in seconds every 1 second before the tracking starts
            if only_do_once and time.time() - start_time <= countdown_seconds:
                countdown = countdown_seconds - int(time.time() - start_time)
                print(f"Set default pose with handle in {countdown} seconds")
                time.sleep(1)
                

            if only_do_once == True and time.time() - start_time > countdown_seconds:
                only_do_once = False
                X_green_start = X_green
                Y_green_start = Y_green
                Z_green_start = Z_green
                X_blue_start = X_blue
                Y_blue_start = Y_blue
                Z_blue_start = Z_blue

            if only_do_once == False:
                delta_X_green = X_green - X_green_start
                delta_Y_green = Y_green_start - Y_green
                delta_Z_green = Z_green_start - Z_green

                delta_X_blue = X_blue - X_blue_start
                delta_Y_blue = Y_blue_start - Y_blue
                delta_Z_blue = Z_blue_start - Z_blue
                  
                Delta_X = (delta_Y_green + delta_Y_blue)/2 # the z coordinate measured by the camera is the x coordinate in the robot CS
                Delta_Y = (delta_Z_green + delta_Z_blue)/2
                Delta_Z = (delta_X_green + delta_X_blue)/2
                Delta_psi_y = -np.arctan2((Y_green - Y_blue), (X_green - X_blue))-np.pi/2 #rotation around x-axis
                Delta_psi_z = np.arctan((Z_green - Z_blue)/ (Y_green - Y_blue)) #rotation around y-axis
                distance = np.sqrt((Z_green - Z_blue)**2 + (X_green - X_blue)**2 + (Y_green - Y_blue)**2)

                delta_target_pose = [Delta_X, Delta_Y, Delta_Z, Delta_psi_y, Delta_psi_z, distance]
                delta_target_pose_list.append(delta_target_pose)

                
                Delta_X_filtered, Delta_Y_filtered, Delta_Z_filtered, Delta_psi_y_filtered, Delta_psi_z_filtered, distance_filtered = delta_target_pose
                
                #print(f"Delta_X_filtered, Delta_Y_filtered, Delta_Z_filtered, Delta_Alpha_filtered, Delta_Beta_filtered, distance_filtered: {int(Delta_X_filtered):>5} {int(Delta_Y_filtered):>5} {int(Delta_Z_filtered):>5} {int(np.rad2deg(Delta_psi_y_filtered)):>5} {int(np.rad2deg(Delta_psi_z_filtered)):>5} {int(distance_filtered):>5} ")
                x_t = 350+Delta_X_filtered * motion_scaling_factor_position # +x_t
                y_t = 180+Delta_Y_filtered * motion_scaling_factor_position # +y_t
                z_t = Delta_Z_filtered * motion_scaling_factor_position # +z_t
                
                psi_y_t = Delta_psi_y_filtered * motion_scaling_factor_orientation # + psi_y_t
                psi_z_t = Delta_psi_z_filtered * motion_scaling_factor_orientation # psi_z_t
                psi_x_t = np.arctan(np.sin(psi_y_t)*np.tan(psi_z_t))

                def cap_angle(angle):
                    return max(min(angle, np.deg2rad(89)), np.deg2rad(-89))

                psi_x_t = cap_angle(psi_x_t)
                psi_y_t = cap_angle(psi_y_t)
                psi_z_t = cap_angle(psi_z_t)

                X_t = [x_t, y_t, z_t, psi_x_t, psi_y_t, psi_z_t]

                #print(f"X_t, Y_t, Z_t, Psi_x_t, Psi_y_t, Psi_z_t: {int(x_t):>5} {int(y_t):>5} {int(z_t):>5} {int(np.rad2deg(psi_x_t)):>5} {int(np.rad2deg(psi_y_t)):>5} {int(np.rad2deg(psi_z_t)):>5}")
                
                q_new = inverse_kinematics_3D_redundant_quaternion(X_t, k, q, error_threshold, design_paras, factor_obj_func=1)
                #q_new = inverse_differential_kinematics_3D_redundant(X_t, k, q, error_threshold, design_paras, factor_obj_func=1)

                if not np.array_equal(q_new, [0,0,0,0,0,0,0]): #[0,0,0,0,0,0,0] is returned from the IKM when there is no convergence
                    update_counter += 1
                    q = q_new
                    fk = forward_kinematics_3D_redundant_quaternion(q, design_paras, resolution = 30)
                    # fk = forward_kinematics_3D_redundant(q, design_paras, resolution = 30)

                    separator = np.ones((masked_image_right.shape[0], 10, 3), dtype=np.uint8) * 255 #Create a white vertical separator
                    combined_image = np.hstack((masked_image_left, separator, masked_image_right, separator, right_array[:,:,:3]))
                    fig = plot_func_3D_video(fk, fig=fig, combined_image=combined_image, update=True, save_video=True, update_counter=update_counter)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                    plt.close('all')  # Close all matplotlib plots
                    send_converted_q_to_arduino([0,0,0,0,0,0,0]) # Send the command to the arduino to stop the motors
                    cv2.destroyAllWindows()
                    zed.close()
                    exit()

    
if __name__ == "__main__":
    retrieve_target_pose()
