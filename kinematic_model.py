import numpy as np
from plotting_numeric_Paper3 import plot_func_3D, plot_error, plot_helix_path_with_orientation, plot_time_boxplot
import time
from scipy.spatial.transform import Rotation
import sympy as sp
#import spacenavigator_adapted
#spacenavigator_adapted.open()



def forward_kinematics_3D_euler(q, resolution = 1):
    alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q #all anlges are in rad
    
    # Design parameters
    len_distal_part = 130 #mm
    len_enddisk = 46 #mm
    len_wrist = 42 #mm
    dist_tendons= 9.7 #mm

    T_list = []

     #it always does the translation first, and then the last point of the translation is rotated about the angle.

    T_list.append(np.array([[1, 0            , 0             , 0      ],
                            [0, np.cos(alpha), -np.sin(alpha), 0      ],
                            [0, np.sin(alpha), np.cos(alpha) , 0      ],
                            [0, 0            , 0             , 1      ]])) 
  
    # transformation matrix for rotation about the z-axis
    def calculate_segment_transform(angle, length):
        return np.array([[np.cos(angle), -np.sin(angle), 0, length],
                         [np.sin(angle), np.cos(angle) , 0, 0     ],
                         [0            , 0             , 1, 0     ],
                         [0            , 0             , 0, 1     ]])
    
    
    
    T_list.append(calculate_segment_transform(theta, 0))

    T_list.append(calculate_segment_transform(np.deg2rad(-90), len_distal_part)) #-90deg to make the robot come out in a perpendicular fashion
    
    def forward_kinematics_CR_3D(l_right, l_left, dist_tendons):
        gamma = ((l_right-l_left) * 180)/(np.pi * dist_tendons) #in deg
        l = (l_right+l_left)/2 
        r = (l * 180/(np.pi * gamma))+1e-4

        kappa = 1/r

        T_CR = np.array([[np.cos(kappa*l),-np.sin(kappa*l), 0, 1/kappa * (np.cos(kappa * l)-1) ],
                        [np.sin(kappa*l) , np.cos(kappa*l), 0, 1/kappa * np.sin(kappa * l)     ],
                        [0               , 0              , 1, 0                               ],
                        [0               , 0              , 0, 1                               ]])
        
        return T_CR

    
    for i in range (resolution):
        T_list.append(forward_kinematics_CR_3D(robotic_length/resolution, (robotic_length+delta_l_niTi)/resolution, dist_tendons))

   
    T_list.append(calculate_segment_transform(np.deg2rad(90), 0))


    T_list.append(np.array([[np.cos(rho_proximal) , 0, np.sin(rho_proximal), len_enddisk],
                            [0                    , 1, 0                   , 0        ],
                            [-np.sin(rho_proximal), 0, np.cos(rho_proximal), 0        ],
                            [0                    , 0, 0                   , 1        ]])) 
            


    T_list.append(np.array([[1, 0                 , 0                  , len_wrist ],
                            [0, np.cos(rho_distal), -np.sin(rho_distal), 0 ],
                            [0, np.sin(rho_distal), np.cos(rho_distal) , 0 ],
                            [0, 0                 , 0                  , 1 ]])) 


    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))

    return T_sum_list


def inverse_kinematics_3D_euler(X_d, k, q, error_threshold, joint_bounds = None):  #difference is that the orientation error is calculated correctly 

    def analytical_jacobian_3D(q):
        h = 0.001
        J_eA = np.zeros((6,6)) #

        fk = forward_kinematics_3D_euler(q)
        #retrieve the X-Y-Z euler angles from the rotation matrix
        position = fk[-1][:3,3]
        R = fk[-1][:3,:3]
        x_angle = np.arctan2(-R[1,2], R[2,2])
        y_angle = np.arctan2( R[0,2], np.sqrt(R[0,0]**2 + R[0,1]**2))
        z_angle = np.arctan2(-R[0,1], R[0,0])
        X_e_q = np.array([position[0], position[1], position[2], x_angle, y_angle, z_angle])

        for i in range(len(q)):
            q_h = q.copy()
            q_h[i] = q[i]+h #i=0 -> derivation w.r.t. q_0, i=1 -> derivation w.r.t. q_1, etc.
            fk_q_h = forward_kinematics_3D_euler(q_h)
            
            #retrieve the X-Y-Z euler angles from the rotation matrix
            position_h = fk_q_h[-1][:3,3]
            R_h = fk_q_h[-1][:3,:3]
            x_angle_h = np.arctan2(-R_h[1,2], R_h[2,2])
            y_angle_h = np.arctan2( R_h[0,2], np.sqrt(R_h[0,0]**2 + R_h[0,1]**2))
            z_angle_h = np.arctan2(-R_h[0,1], R_h[0,0])
            X_e_q_h = np.array([position_h[0], position_h[1], position_h[2], x_angle_h, y_angle_h, z_angle_h])
        
            J_eA[:,i] = (X_e_q_h - X_e_q) / h

        return J_eA


    def retrieve_zyx_euler_angles(R):
        # get R matrix 3x3
        # return zyx euler angles
        psi_z = np.arctan2( R[1,0], R[0,0])
        psi_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        psi_x = np.arctan2( R[2,1], R[2,2])
        return np.array([psi_x, psi_y, psi_z])
    
    #fk_start is the fk of the start configuration of the robot, i.e. the fk belonging to the original q 
    fk_start = forward_kinematics_3D_euler(q)
    
    x_position = fk_start[-1][0,3]
    y_position = fk_start[-1][1,3]
    z_position = fk_start[-1][2,3]
    R_actual = fk_start[-1][:3,:3]
    euler_angle_actual = retrieve_zyx_euler_angles(R_actual)

    X_e = np.array([x_position, y_position, z_position, euler_angle_actual[0], euler_angle_actual[1], euler_angle_actual[2]]) # X_e = current pose parameterization of the end effector 
    
    psi_x_des = X_d[3]
    psi_y_des = X_d[4]
    psi_z_des = X_d[5]

    R_z_des = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                        [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                        [0                ,  0                 ,1]])
    R_y_des = np.array([[np.cos(psi_y_des) , 0, np.sin(psi_y_des)],
                        [0                 , 1,                 0],
                        [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
    R_x_des = np.array([[1, 0                ,                  0],
                        [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                        [0, np.sin(psi_x_des), np.cos(psi_x_des)]])
    
    R_des = R_z_des @ R_y_des @ R_x_des #this order is right for intrinsic (i.e., local) zyx euler angles
    delta_R = R_des @ R_actual.T
    delta_angle = retrieve_zyx_euler_angles(delta_R)

    parameterized_pose_error = X_d-X_e
    parameterized_pose_error[3:] = delta_angle
    error_list = [np.linalg.norm(parameterized_pose_error)]
    counter = 1
    start_time = time.time()

    fk_list = [fk_start]


    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J_eA = analytical_jacobian_3D(q)
        
        q = q + k* np.dot(np.linalg.pinv(J_eA), parameterized_pose_error)
        
        # Calculate new endeffector pose X_e
        fk_new = forward_kinematics_3D_euler(q)
        
        x_position = fk_new[-1][0,3]
        y_position = fk_new[-1][1,3]
        z_position = fk_new[-1][2,3]
        R_actual = fk_new[-1][:3,:3]
        euler_angle_actual = retrieve_zyx_euler_angles(R_actual)

        X_e = np.array([x_position, y_position, z_position, euler_angle_actual[0], euler_angle_actual[1], euler_angle_actual[2]])
        
        delta_R = R_des @ R_actual.T
        delta_angle = retrieve_zyx_euler_angles(delta_R)

        parameterized_pose_error = X_d-X_e
        parameterized_pose_error[3:] = delta_angle
        error_list.append(np.linalg.norm(parameterized_pose_error))
        
        counter = counter + 1
        fk_list.append(fk_new)
        
        if counter == 1000:
            print("Counter reached 1000 -> exit inverse differential kinematics")
            plot_error(error_list, error_threshold, k)
                 
            return 0
    
    convergence_time = round(time.time()-start_time, 4)
    print(f"Minimization of inverse differential kinematics took {convergence_time} seconds")
    #plot_error(error_list, error_threshold, k)

                
    return q





def forward_kinematics_3D_redundant_symbolic(symbols, resolution=1):
    (stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal, len_distal_part, len_enddisk, len_wrist, dist_tendons) = symbols

    
    # Identity Matrix for Initial Transformation
    T_list = [sp.eye(4)]
    
    # Transformation for Stem Length and Alpha Rotation
    T_list.append(sp.Matrix([
        [1, 0, 0, stem_length],
        [0, sp.cos(alpha), -sp.sin(alpha), 0],
        [0, sp.sin(alpha), sp.cos(alpha), 0],
        [0, 0, 0, 1]
    ]))
    
    # Function for a Z-axis Rotation Transformation
    def calculate_segment_transform(angle, length):
        return sp.Matrix([
            [sp.cos(angle), -sp.sin(angle), 0, length],
            [sp.sin(angle), sp.cos(angle), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # Apply Theta Rotation
    T_list.append(calculate_segment_transform(theta, 0))
    
    # Apply -90-degree Rotation and Distal Part Extension
    T_list.append(calculate_segment_transform(sp.rad(-90), len_distal_part))
    
    # Function for Continuum Robot Forward Kinematics
    def forward_kinematics_CR_3D(l_right, l_left, dist_tendons):
        gamma = ((l_right - l_left) * 180) / (sp.pi * dist_tendons)  # in degrees
        l = (l_right + l_left) / 2
        if gamma == 0:
            gamma = 1e-7
        r = (l * 180) / ((sp.pi * gamma))
        kappa = 1 / r
        
        T_CR = sp.Matrix([
            [sp.cos(kappa * l), -sp.sin(kappa * l), 0, 1 / kappa * (sp.cos(kappa * l) - 1)],
            [sp.sin(kappa * l), sp.cos(kappa * l), 0, 1 / kappa * sp.sin(kappa * l)],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        
        return T_CR
    
    # Add Continuum Robot Transformations
    for i in range(resolution):
        T_list.append(forward_kinematics_CR_3D(robotic_length / resolution, (robotic_length + delta_l_niTi) / resolution, dist_tendons))
    
    # Apply 90-degree Rotation
    T_list.append(calculate_segment_transform(sp.rad(90), 0))
    
    # Apply Proximal Wrist Rotation
    T_list.append(sp.Matrix([
        [sp.cos(rho_proximal), 0, sp.sin(rho_proximal), len_enddisk],
        [0, 1, 0, 0],
        [-sp.sin(rho_proximal), 0, sp.cos(rho_proximal), 0],
        [0, 0, 0, 1]]))
    
    # Apply Distal Wrist Rotation
    T_list.append(sp.Matrix([
        [1, 0, 0, len_wrist],
        [0, sp.cos(rho_distal), -sp.sin(rho_distal), 0],
        [0, sp.sin(rho_distal), sp.cos(rho_distal), 0],
        [0, 0, 0, 1]]))
    


    # Compute Cumulative Transformations
    T_sum_list = [T_list[0]]
    for i in range(1, len(T_list)):
        T_sum_list.append(T_sum_list[-1] * T_list[i])
    
    return T_sum_list





def forward_kinematics_3D_redundant_euler(q, design_paras, resolution = 1):
    stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q #all anlges are in rad
    
    len_distal_part, len_enddisk, len_wrist, dist_tendons = design_paras
    

    T_list = [np.eye(4)] #np.eye(4) makes sure that the plotting starts at the CS center (0,0,0)


    #it always does the translation first, and then the last point of the translation is rotated about the angle.
    T_list.append(np.array([[1, 0            , 0             , stem_length],
                            [0, np.cos(alpha), -np.sin(alpha), 0          ],
                            [0, np.sin(alpha), np.cos(alpha) , 0          ],
                            [0, 0            , 0             , 1          ]])) 
  
    # transformation matrix for rotation about the z-axis
    def calculate_segment_transform(angle, length):
        return np.array([[np.cos(angle), -np.sin(angle), 0, length],
                         [np.sin(angle), np.cos(angle) , 0, 0     ],
                         [0            , 0             , 1, 0     ],
                         [0            , 0             , 0, 1     ]])
    
    
    T_list.append(calculate_segment_transform(theta, 0))

    T_list.append(calculate_segment_transform(np.deg2rad(-90), len_distal_part)) #-90deg to make the robot come out in a perpendicular fashion
    
    def forward_kinematics_CR_3D(l_right, l_left, dist_tendons):
        gamma = ((l_right-l_left) * 180)/(np.pi * dist_tendons) #in deg
        l = (l_right+l_left)/2 
        r = (l * 180)/((np.pi * gamma)+1e-5)

        kappa = 1/r

        T_CR = np.array([[np.cos(kappa*l),-np.sin(kappa*l), 0, 1/kappa * (np.cos(kappa * l)-1) ],
                        [np.sin(kappa*l) , np.cos(kappa*l), 0, 1/kappa * np.sin(kappa * l)     ],
                        [0               , 0              , 1, 0                               ],
                        [0               , 0              , 0, 1                               ]])
        
        return T_CR

    
    for i in range (resolution):
        T_list.append(forward_kinematics_CR_3D(robotic_length/resolution, (robotic_length+delta_l_niTi)/resolution, dist_tendons))

   
    T_list.append(calculate_segment_transform(np.deg2rad(90), 0))


    T_list.append(np.array([[np.cos(rho_proximal) , 0, np.sin(rho_proximal), len_enddisk],
                            [0                    , 1, 0                   , 0        ],
                            [-np.sin(rho_proximal), 0, np.cos(rho_proximal), 0        ],
                            [0                    , 0, 0                   , 1        ]])) 
            


    T_list.append(np.array([[1, 0                 , 0                  , len_wrist ],
                            [0, np.cos(rho_distal), -np.sin(rho_distal), 0 ],
                            [0, np.sin(rho_distal), np.cos(rho_distal) , 0 ],
                            [0, 0                 , 0                  , 1 ]])) 


    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))

    return T_sum_list


def inverse_kinematics_3D_redundant_euler(X_d, k, q, error_threshold, design_paras, factor_obj_func=0):  #difference is that the orientation error is calculated correctly 

    def analytical_jacobian_3D_redundant(q):
        h = 0.001
        J_eA = np.zeros((6,7)) #

        fk = forward_kinematics_3D_redundant_euler(q, design_paras)
        #retrieve the X-Y-Z euler angles from the rotation matrix
        position = fk[-1][:3,3]
        R = fk[-1][:3,:3]
        x_angle = np.arctan2(-R[1,2], R[2,2])
        y_angle = np.arctan2( R[0,2], np.sqrt(R[0,0]**2 + R[0,1]**2))
        z_angle = np.arctan2(-R[0,1], R[0,0])
        X_e_q = np.array([position[0], position[1], position[2], x_angle, y_angle, z_angle])

        for i in range(len(q)):
            q_h = q.copy()
            q_h[i] = q[i]+h #i=0 -> derivation w.r.t. q_0, i=1 -> derivation w.r.t. q_1, etc.
            fk_q_h = forward_kinematics_3D_redundant_euler(q_h, design_paras)
            
            #retrieve the X-Y-Z euler angles from the rotation matrix
            position_h = fk_q_h[-1][:3,3]
            R_h = fk_q_h[-1][:3,:3]
            x_angle_h = np.arctan2(-R_h[1,2], R_h[2,2])
            y_angle_h = np.arctan2( R_h[0,2], np.sqrt(R_h[0,0]**2 + R_h[0,1]**2))
            z_angle_h = np.arctan2(-R_h[0,1], R_h[0,0])
            X_e_q_h = np.array([position_h[0], position_h[1], position_h[2], x_angle_h, y_angle_h, z_angle_h])
        
            J_eA[:,i] = (X_e_q_h - X_e_q) / h

        return J_eA

    def dist_sq_penalty(q_val, q_min, q_max, k_out):
        """
        Returns the penalty and its derivative w.r.t. q_val
        """
        if q_val < q_min:
            pen = k_out*(q_min - q_val)**2
            dpen_dq = -2*k_out*(q_min - q_val)
        elif q_val > q_max:
            pen = k_out*(q_val - q_max)**2
            dpen_dq = 2*k_out*(q_val - q_max)
        else:
            pen = 0.0
            dpen_dq = 0.0
        return pen, dpen_dq

    def objective_H_and_gradient(q):
        # returns the value of the objective function "val_H" and its gradient "grad_H"

        # Suppose q is [q0, q1, q2, q3, q4, q5, q6]
        #w0 = 1 # maximize q0 (i.e. the stem length)
        k_out = 10#20.0        # penalty factor

        # 1 "Maximize q0"
        #val_H = w0*q[0]
        
        # store partial derivatives in grad_H
        grad_H = np.zeros_like(q)

        # a gradient from "w0*q0 + w2*q2":
        grad_H[0] += w0
        #grad_H[2] += w2

        # penatly for stem_length if outside [50, 400]
        q0_min, q0_max = 40, 400
        pen_q0, dpen_q0 = dist_sq_penalty(q[0], q0_min, q0_max, k_out)
        val_H -= pen_q0
        grad_H[0] -= dpen_q0

        # no penalty for alpha, as alpha is handled in the interpret_results function
        #q1_min, q1_max = np.deg2rad(-179), np.deg2rad(180)
        #pen_q1, dpen_q1 = dist_sq_penalty(q[1], q1_min, q1_max, k_out)
        #val_H -= pen_q1
        #grad_H[1] -= dpen_q1  

        # penalty for robotic_len if outside [30mm, 150mm]
        q2_min, q2_max = 30, 150
        pen_q2, dpen_q2 = dist_sq_penalty(q[2], q2_min, q2_max, k_out)
        val_H -= pen_q2
        grad_H[2] -= dpen_q2

        # penalty for theta if outside [0 deg, 60 deg]
        q3_min, q3_max = 0.0, np.deg2rad(60)
        pen_q3, dpen_q3 = dist_sq_penalty(q[3], q3_min, q3_max, k_out)
        val_H -= pen_q3
        grad_H[3] -= dpen_q3 

        # penalty for delta_l_niti if outside [-20mm, 30mm]
        q4_min, q4_max = -20, 30
        pen_q4, dpen_q4 = dist_sq_penalty(q[4], q4_min, q4_max, k_out)
        val_H -= pen_q4
        grad_H[4] -= dpen_q4

        # penalty for rho_prox if outside [-80 deg, 80 deg]
        q5_min, q5_max = np.deg2rad(-80), np.deg2rad(80)
        pen_q5, dpen_q5 = dist_sq_penalty(q[5], q5_min, q5_max, k_out)
        val_H -= pen_q5
        grad_H[5] -= dpen_q5

        # no penalty for rho_dist
        # q6_min, q6_max = np.deg2rad(-90), np.deg2rad(90)
        # pen_q6, dpen_q6 = dist_sq_penalty(q[6], q6_min, q6_max, k_out)
        # val_H -= pen_q6
        # grad_H[6] -= dpen_q6

        return val_H, grad_H


    def retrieve_zyx_euler_angles(R):
        # get R matrix 3x3
        # return zyx euler angles
        psi_z = np.arctan2( R[1,0], R[0,0])
        psi_y = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
        psi_x = np.arctan2( R[2,1], R[2,2])
        return np.array([psi_x, psi_y, psi_z])
    
    def interpret_results(q):
        
        def wrap_angle(angle): #angle in rad
            return (angle + np.pi) % (2*np.pi) - np.pi
        
        stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q[0], wrap_angle(q[1]), q[2], wrap_angle(q[3]), q[4], wrap_angle(q[5]), wrap_angle(q[6])

        #these conditions check if the inverse kinematic solution has a negative theta. mechanically this is not possible, so we need to rotate the entire robot about alpha and then use positive theta
        if theta < 0 and alpha > 0: 
            alpha -= np.pi
            theta = -theta
            delta_l_niTi = -delta_l_niTi
            rho_proximal = -rho_proximal
            rho_distal = -rho_distal
        if theta < 0 and alpha < 0:
            alpha += np.pi
            theta = -theta
            delta_l_niTi = -delta_l_niTi
            rho_proximal = -rho_proximal
            rho_distal = -rho_distal

        return [stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal]
  

    #fk_start is the fk of the start configuration of the robot, i.e. the fk belonging to the original q 
    fk_start = forward_kinematics_3D_redundant_euler(q, design_paras)
    
    x_position = fk_start[-1][0,3]
    y_position = fk_start[-1][1,3]
    z_position = fk_start[-1][2,3]
    R_actual = fk_start[-1][:3,:3]
    euler_angle_actual = retrieve_zyx_euler_angles(R_actual)

    X_e = np.array([x_position, y_position, z_position, euler_angle_actual[0], euler_angle_actual[1], euler_angle_actual[2]]) # X_e = current pose parameterization of the end effector 
    
    psi_x_des = X_d[3]
    psi_y_des = X_d[4]
    psi_z_des = X_d[5]

    R_z_des = np.array([[np.cos(psi_z_des), -np.sin(psi_z_des), 0],
                        [np.sin(psi_z_des),  np.cos(psi_z_des), 0],
                        [0                ,  0                 ,1]])
    R_y_des = np.array([[np.cos(psi_y_des) , 0, np.sin(psi_y_des)],
                        [0                 , 1,                 0],
                        [-np.sin(psi_y_des), 0, np.cos(psi_y_des)]])
    R_x_des = np.array([[1, 0                ,                  0],
                        [0, np.cos(psi_x_des), -np.sin(psi_x_des)],
                        [0, np.sin(psi_x_des), np.cos(psi_x_des)]])
    
    R_des = R_z_des @ R_y_des @ R_x_des #this order is right for intrinsic (i.e., local) zyx euler angles
    delta_R = R_des @ R_actual.T
    delta_angle = retrieve_zyx_euler_angles(delta_R)

    parameterized_pose_error = X_d-X_e
    parameterized_pose_error[3:] = delta_angle
    error_list = [np.linalg.norm(parameterized_pose_error)]
    counter = 1
    start_time = time.time()

    fk_list = [fk_start]


    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J_eA = analytical_jacobian_3D_redundant(q)
        _, gradH = objective_H_and_gradient(q)
        q = q + k* np.dot(np.linalg.pinv(J_eA), parameterized_pose_error) + factor_obj_func * np.dot(np.eye(len(q)) - np.dot(np.linalg.pinv(J_eA), J_eA), gradH) # np.array([0,0,-1,-1,0,0,0]
        
        # Calculate new endeffector pose X_e
        fk_new = forward_kinematics_3D_redundant_euler(q, design_paras)
        
        x_position = fk_new[-1][0,3]
        y_position = fk_new[-1][1,3]
        z_position = fk_new[-1][2,3]
        R_actual = fk_new[-1][:3,:3]
        euler_angle_actual = retrieve_zyx_euler_angles(R_actual)

        X_e = np.array([x_position, y_position, z_position, euler_angle_actual[0], euler_angle_actual[1], euler_angle_actual[2]])
        
        delta_R = R_des @ R_actual.T
        delta_angle = retrieve_zyx_euler_angles(delta_R)

        parameterized_pose_error = X_d-X_e
        parameterized_pose_error[3:] = delta_angle
        error_list.append(np.linalg.norm(parameterized_pose_error))
        
        counter = counter + 1
        fk_list.append(fk_new)
        
        if counter == 1000:
            print("Counter reached 1000 -> exit inverse differential kinematics")
            plot_error(error_list, error_threshold, k)
                 
            return np.array([0,0,0,0,0,0,0])
    
    convergence_time = round(time.time()-start_time, 4)
    #print(f"Minimization of inverse differential kinematics took {convergence_time} seconds")
    #plot_error(error_list, error_threshold, k)

    q_final = interpret_results(q)

    return q_final






def forward_kinematics_3D_redundant_quaternion(q, design_paras, resolution = 1):

    def T_quat(rot_axis, theta, trans_vec):
        """
        Generates a transformation matrix from a quaternion for a given axis ('x', 'y', or 'z') and angle (radians) and the translation vector.
        
        Parameters:
            axis (str): The axis to rotate about ('x', 'y', or 'z').
            theta (float): The rotation angle in radians.
            trans_vec [trans in x, trans in y, trans in z], trranslation in mm
            
        Returns:
            np.ndarray: A 4x4 transformation matrix.
        """
        if rot_axis == 'x':
            xi = [np.cos(theta/2), np.sin(theta/2), 0, 0]  # Quaternion (w, x, y, z)
        elif rot_axis == 'y':
            xi = [np.cos(theta/2), 0, np.sin(theta/2), 0]  # Quaternion (w, x, y, z)
        elif rot_axis == 'z':
            xi = [np.cos(theta/2), 0, 0, np.sin(theta/2)]  # Quaternion (w, x, y, z)
        else:
            raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

        # Extract quaternion components
        xi0, xi1, xi2, xi3 = xi  

        # Compute rotation matrix from quaternion
        T = np.array([
            [1 - 2 * (xi2**2 + xi3**2), 2 * (xi1*xi2 - xi0*xi3)   , 2 * (xi1*xi3 + xi0*xi2)   , trans_vec[0]],
            [2 * (xi1*xi2 + xi0*xi3)  , 1 - 2 * (xi1**2 + xi3**2) , 2 * (xi2*xi3 - xi0*xi1)   , trans_vec[1]],
            [2 * (xi1*xi3 - xi0*xi2)  , 2 * (xi2*xi3 + xi0*xi1)   , 1 - 2 * (xi1**2 + xi2**2) , trans_vec[2]],
            [0                        , 0                         , 0                         , 1           ]
        ])

        return T

    def forward_kinematics_CR_3D_quat(l_right, l_left, dist_tendons):
            """
            Computes the forward kinematics of a continuum robot in 3D using quaternions.

            Parameters:
                l_right (float): Length of the right tendon.
                l_left (float): Length of the left tendon.
                dist_tendons (float): Distance between tendons.

            Returns:
                np.ndarray: 4x4 Transformation matrix.
            """
            gamma = ((l_right - l_left) * 180) / (np.pi * dist_tendons)  # in degrees
            l = (l_right + l_left) / 2  
            r = (l * 180 / (np.pi * gamma)) + 1e-4  # Avoid division by zero

            kappa = 1 / r
            theta = kappa * l  # Total bending angle

            # Rotation is about the Z-axis in the Frenet-Serret frame
            rot_axis = 'z'
            
            # Translation vector along X-axis in the local frame
            trans_vec = [1/kappa * (np.cos(theta) - 1), 1/kappa * np.sin(theta), 0]

            # Compute transformation matrix using quaternions
            T_CR = T_quat(rot_axis, theta, trans_vec)
            
            return T_CR


    stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q #all anlges are in rad
    
    len_distal_part, len_enddisk, len_wrist, dist_tendons = design_paras


    T_list = [np.eye(4)]
    T_list.append(T_quat('x', alpha, [stem_length,0,0]))
    T_list.append(T_quat('z', theta, [0,0,0]))
    T_list.append(T_quat('z', np.deg2rad(-90), [len_distal_part,0,0]))
    
    
    for i in range (resolution):
        T_list.append(forward_kinematics_CR_3D_quat(robotic_length/resolution, (robotic_length+delta_l_niTi)/resolution, dist_tendons))

    T_list.append(T_quat('z', np.deg2rad(90), [0,0,0]))


    T_list.append(T_quat('y', rho_proximal, [len_enddisk,0,0]))
    T_list.append(T_quat('x', rho_distal, [len_wrist,0,0]))
            

    T_sum_list = [T_list[0]]
    for i in range(1,len(T_list)):
        T_sum_list.append(np.dot(T_sum_list[-1],T_list[i]))

    return T_sum_list
    

def inverse_kinematics_3D_redundant_quaternion(X_d, k, q, error_threshold, design_paras, factor_obj_func=0, create_mp4=False):
    
    def analytical_jacobian_3D_redundant_quat(q):
        h = 0.001
        J_eA = np.zeros((7, 7))  # Now 7x7 because of quaternion representation

        fk = forward_kinematics_3D_redundant_quaternion(q, design_paras)
        position = fk[-1][:3,3]
        R = fk[-1][:3,:3]
        quat_actual = Rotation.from_matrix(R).as_quat()  # Convert rotation matrix to quaternion
        X_e_q = np.concatenate((position, quat_actual))

        for i in range(len(q)):
            q_h = q.copy()
            q_h[i] += h  
            fk_q_h = forward_kinematics_3D_redundant_quaternion(q_h, design_paras)
        
            position_h = fk_q_h[-1][:3,3]
            R_h = fk_q_h[-1][:3,:3]
            quat_h = Rotation.from_matrix(R_h).as_quat()
            X_e_q_h = np.concatenate((position_h, quat_h))
    
            J_eA[:, i] = (X_e_q_h - X_e_q) / h

        return J_eA
        
    def quaternion_from_rotation_matrix(R):
        """Convert a 3x3 rotation matrix into a quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    def quaternion_inverse(q):
        """Compute the inverse of a quaternion."""
        w, x, y, z = q
        norm_sq = w**2 + x**2 + y**2 + z**2  # Compute squared norm
        return np.array([w, -x, -y, -z]) / norm_sq  # Return inverse

    def quaternion_multiply(q1, q2):
        """Multiply two quaternions (Hamilton product)."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2])

    def quaternion_from_euler(roll, pitch, yaw):
        """Convert intrinsic ZYX Euler angles to a quaternion (w, x, y, z)."""
        cr = np.cos(roll / 2)
        sr = np.sin(roll / 2)
        cp = np.cos(pitch / 2)
        sp = np.sin(pitch / 2)
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.array([w, x, y, z])

    def quaternion_error(q_d, q_e):
        """Computes quaternion orientation error as a rotation vector (3D)."""
        q_e_inv = quaternion_inverse(q_e)  
        q_err = quaternion_multiply(q_d, q_e_inv)  

        angle = 2 * np.arccos(np.clip(q_err[0], -1.0, 1.0))  
        if np.abs(angle) < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        
        axis = q_err[1:] / np.sin(angle / 2)
        return angle * axis  

    def dist_sq_penalty(q_val, q_min, q_max, k_out):
        """
        Returns the penalty and its derivative w.r.t. q_val
        """
        if q_val < q_min:
            pen = k_out*(q_min - q_val)**2
            dpen_dq = -2*k_out*(q_min - q_val)
        elif q_val > q_max:
            pen = k_out*(q_val - q_max)**2
            dpen_dq = 2*k_out*(q_val - q_max)
        else:
            pen = 0.0
            dpen_dq = 0.0
        return pen, dpen_dq

    def objective_H_and_gradient(q):
        # returns the value of the objective function "val_H" and its gradient "grad_H"

        # Suppose q is [q0, q1, q2, q3, q4, q5, q6]
        #w0 = 1 # maximize q0 (i.e. the stem length)
        w2 = -1 # minimize q2 (i.e. rob_len)
        k_out = 5#10.0        # penalty factor

        # 1 "Maximize q2"
        val_H = w2*q[0] 
        
        # store partial derivatives in grad_H
        grad_H = np.zeros_like(q)

        # a gradient from "w0*q0 + w2*q2":
        #grad_H[0] += w0
        grad_H[3] += w2

        # penatly for stem_length if outside [50, 400]
        q0_min, q0_max = 40, 400
        pen_q0, dpen_q0 = dist_sq_penalty(q[0], q0_min, q0_max, k_out)
        val_H -= pen_q0
        grad_H[0] -= dpen_q0

        # no penalty for alpha, as alpha is handled in the interpret_results function
        #q1_min, q1_max = np.deg2rad(-179), np.deg2rad(180)
        #pen_q1, dpen_q1 = dist_sq_penalty(q[1], q1_min, q1_max, k_out)
        #val_H -= pen_q1
        #grad_H[1] -= dpen_q1  

        # penalty for robotic_len if outside [30mm, 150mm]
        q2_min, q2_max = 30, 150
        pen_q2, dpen_q2 = dist_sq_penalty(q[2], q2_min, q2_max, k_out)
        val_H -= pen_q2
        grad_H[2] -= dpen_q2

        # penalty for theta if outside [0 deg, 60 deg]
        q3_min, q3_max = 0.0, np.deg2rad(50)
        pen_q3, dpen_q3 = dist_sq_penalty(q[3], q3_min, q3_max, k_out)
        val_H -= pen_q3
        grad_H[3] -= dpen_q3 

        # penalty for delta_l_niti if outside [-20mm, 30mm]
        q4_min, q4_max = -20, 30
        pen_q4, dpen_q4 = dist_sq_penalty(q[4], q4_min, q4_max, k_out)
        val_H -= pen_q4
        grad_H[4] -= dpen_q4

        # penalty for rho_prox if outside [-80 deg, 80 deg]
        q5_min, q5_max = np.deg2rad(-80), np.deg2rad(80)
        pen_q5, dpen_q5 = dist_sq_penalty(q[5], q5_min, q5_max, k_out)
        val_H -= pen_q5
        grad_H[5] -= dpen_q5

        # no penalty for rho_dist
        # q6_min, q6_max = np.deg2rad(-90), np.deg2rad(90)
        # pen_q6, dpen_q6 = dist_sq_penalty(q[6], q6_min, q6_max, k_out)
        # val_H -= pen_q6
        # grad_H[6] -= dpen_q6

        return val_H, grad_H

    def interpret_results(q):
        
        def wrap_angle(angle): #angle in rad
            return (angle + np.pi) % (2*np.pi) - np.pi
        
        stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q[0], wrap_angle(q[1]), q[2], wrap_angle(q[3]), q[4], wrap_angle(q[5]), wrap_angle(q[6])

        #these conditions check if the inverse kinematic solution has a negative theta. mechanically this is not possible, so we need to rotate the entire robot about alpha and then use positive theta
        if theta < 0 and alpha > 0: 
            alpha -= np.pi
            theta = -theta
            delta_l_niTi = -delta_l_niTi
            rho_proximal = -rho_proximal
            rho_distal = -rho_distal
        if theta < 0 and alpha < 0:
            alpha += np.pi
            theta = -theta
            delta_l_niTi = -delta_l_niTi
            rho_proximal = -rho_proximal
            rho_distal = -rho_distal

        return [stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal]
  

    fk_start = forward_kinematics_3D_redundant_quaternion(q, design_paras)
    x_position, y_position, z_position = fk_start[-1][:3,3]
    R_actual = fk_start[-1][:3,:3]
    quat_actual = quaternion_from_rotation_matrix(R_actual)

    pos_desired = X_d[:3]
    quat_desired = quaternion_from_euler(X_d[3], X_d[4], X_d[5]) 

    X_e = np.concatenate((np.array([x_position, y_position, z_position]), quat_actual))
    pos_error = pos_desired - X_e[:3]

    quat_error = quaternion_error(quat_desired, X_e[3:])
    parameterized_pose_error = np.concatenate((pos_error, quat_error))

    error_list = [np.linalg.norm(parameterized_pose_error)]
    counter = 1
    fk_list = [fk_start]

    while np.linalg.norm(parameterized_pose_error) > error_threshold:
        J_eA = analytical_jacobian_3D_redundant_quat(q) # returns 7x7 Jacobian matrix, rows being the derivatives of parameterized end effector pose (x, y, z, xi1, xi2, xi3, xi4) and columns being the derivatives of the generalized joint coordinates (q1 = stem length, q2 = alpha, q3 = ...)  
        J_eA = J_eA[:6, :]  # Only consider the first 6 rows (i.e., position part of the Jacobian)
        _, gradH = objective_H_and_gradient(q)
        q = q + k * np.dot(np.linalg.pinv(J_eA), parameterized_pose_error) + factor_obj_func * np.dot(np.eye(len(q)) - np.dot(np.linalg.pinv(J_eA), J_eA), gradH) #  np.array([0,0,0,-0.5,0,0,0])

        fk_new = forward_kinematics_3D_redundant_quaternion(q, design_paras)
        x_position, y_position, z_position = fk_new[-1][:3,3]
        R_actual = fk_new[-1][:3,:3]
        quat_actual = quaternion_from_rotation_matrix(R_actual)

        X_e = np.concatenate((np.array([x_position, y_position, z_position]), quat_actual))
        
        pos_error = pos_desired - X_e[:3]
        quat_error = quaternion_error(quat_desired, X_e[3:])
        parameterized_pose_error = np.concatenate((pos_error, quat_error))

        error_list.append(np.linalg.norm(parameterized_pose_error))
        counter += 1
        fk_list.append(fk_new)

        if counter == 1000:
            print("Counter reached 1000 -> exit inverse differential kinematics -----------------------------------------------------------------------------------------------")
            return np.array([0,0,0,0,0,0,0])

    q_final = interpret_results(q)
    return q_final


def convert_q_to_motor_commands(q):
        
        def cap_value(value, min_value, max_value, name_of_value):
            capped_value = np.clip(value, min_value, max_value)
            if capped_value != value:
                print(name_of_value + " was capped to " + str(capped_value))
            return int(capped_value)
        
        
        #dist_rot_axis_2_rob_center_line = 11.02
        dist_NiTi_2_connectors = 9.7 #mm
        dist_rot_axis_2_niti = 4.7 #mm
        dist_rot_axis_2_connectors = dist_NiTi_2_connectors + dist_rot_axis_2_niti #mm
        
        dist_connectors_2_upper_tendon = 2.45 #mm
        dist_rot_axis_2_upper_tendon = 12.3 #mm
        radius_tendon_motor = 14 #mm
        radius_tendon_endeffector = 4.5 #mm

        microsteps = 1600 #steps per revolution of the stepper motor driver
        deg_per_step = 360 / microsteps #if microsteps = 1600, then deg_per_step = 0.225

        stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal = q
        
        #convert q[0] in mm translation into degrees, 5mm pitch (gewindesteigung) -> 1 full rotation with 360 deg = 5mm
        conversion_deg_per_mm = 360/5
        rotation_motor1_in_deg = conversion_deg_per_mm * stem_length 
        rotation_motor1_in_steps = rotation_motor1_in_deg / deg_per_step
        rotation_motor1_in_steps = cap_value(rotation_motor1_in_steps, 0, 50000, "stem length (motor 1)")
        
        #convert q[1] in rad alpha into degrees, small gear diameter = 20mm, large gear diameter = 120mm
        gear_ratio = 120 / 20
        rotation_motor2_in_deg = gear_ratio * np.rad2deg(alpha)
        rotation_motor2_in_steps = rotation_motor2_in_deg / deg_per_step #calculation seems right
        rotation_motor2_in_steps = cap_value(rotation_motor2_in_steps, -2400, 2400, "alpha (motor2)")
        
        #convert robotic length q[2] in mm to rot in deg and compensate due to theta
        gear_radius = 15
        rotation_motor3_in_deg = np.rad2deg((robotic_length + (theta*dist_rot_axis_2_connectors))/gear_radius)
        rotation_motor3_in_steps = rotation_motor3_in_deg / deg_per_step
        rotation_motor3_in_steps = cap_value(rotation_motor3_in_steps, 0, 2700, "rob length (motor3)")

        #convert q[3], theta for the rack pinion mechanism of the wrist joint into a rotation in deg of the motor
        #pinion_radius = gear_on_motor_radius = 15 --> no transmission, so 1:1
        rotation_motor4_in_deg = np.rad2deg(theta)
        rotation_motor4_in_steps = rotation_motor4_in_deg / deg_per_step
        rotation_motor4_in_steps = cap_value(rotation_motor4_in_steps, 0, 190, "theta AAU (motor4)")

        #convert q[4] in mm delta_l_niTi into degrees, upward bending (delta_l_niti < 0) -> positive rotation_motor5, downward bending (delta_l_niti > 0) -> negative rotation_motor5
        rotation_motor5_in_deg = np.rad2deg((delta_l_niTi + (theta*dist_rot_axis_2_niti))/gear_radius)
        rotation_motor5_in_steps = -rotation_motor5_in_deg / deg_per_step
        rotation_motor5_in_steps = cap_value(rotation_motor5_in_steps, -600, 300, "delta_l_niti (motor5)")

        #convert rho proximal (controlled by the tendons) into rotation of the motor
        gamma = np.abs(((delta_l_niTi + robotic_length) - robotic_length)/dist_NiTi_2_connectors) #gamma is the HHR bending angle
        radius_connectors = robotic_length / gamma
        if delta_l_niTi < 0: #upward bending            
            radius_upper_tendon = radius_connectors - dist_connectors_2_upper_tendon
        else:# delta_l_niTi > 0: downward bending
            radius_upper_tendon = radius_connectors + dist_connectors_2_upper_tendon
        length_upper_tendon_bent = gamma * radius_upper_tendon
        delta_length_upper_tendon = length_upper_tendon_bent - robotic_length #delta length between straight and bent HRR, <0 for upward bending
        # print("rho_proximal*radius_tendon_endeffector: ", rho_proximal*radius_tendon_endeffector)
        # print("delta_length_upper_tendon: ", delta_length_upper_tendon)
        # print("delta_l_niTi: ", delta_l_niTi)
        # print("--------------------")
        rotation_motor6_in_deg = np.rad2deg((rho_proximal*radius_tendon_endeffector + delta_length_upper_tendon) / radius_tendon_motor) #the upper wire goes to the left EE side 
        rotation_motor6_in_steps = rotation_motor6_in_deg / deg_per_step
        rotation_motor6_in_steps = cap_value(rotation_motor6_in_steps, -200, 200, "rho_prox (motor6)")

        #camera rotation in deg done in python 
        virtual_motor7_in_deg = rho_distal
        
        return np.array([rotation_motor1_in_steps, rotation_motor2_in_steps, rotation_motor3_in_steps, rotation_motor4_in_steps, rotation_motor5_in_steps, rotation_motor6_in_steps, virtual_motor7_in_deg])





if __name__ == "__main__":

    # Design parameters
    len_distal_part = 130 #mm
    len_enddisk = 46 #mm
    len_wrist = 42 #mm
    dist_tendons= 9.7 #mm
    design_paras = [len_distal_part, len_enddisk, len_wrist, dist_tendons]

    plot_forward_kinematics_3D = 0
    forward_kinematics_3D_symbolic = 0
    plot_Ik_3D = 0
    plot_Ik_3d_redundant = 0
    plot_Ik_3D_redundant_quat = 0
    plot_helix_trajectory  = 0
    comparison_Ik_3D_redundant_euler_and_quat = 0
    

    if plot_forward_kinematics_3D:

        stem_length = 100
        alpha = np.deg2rad(40) # rotation of the robot along the shaft's center point
        robotic_length = 150 # #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = np.deg2rad(60) # positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = 8 # mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = np.deg2rad(40) # wirst joint
        rho_distal = np.arctan((-np.sin(alpha)*np.cos(rho_proximal)-np.sin(rho_proximal)*np.sin((delta_l_niTi/dist_tendons) - theta)*np.cos(alpha))/(np.cos(alpha)*np.cos((delta_l_niTi/dist_tendons) - theta)))

        q = [stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal]
        fk = forward_kinematics_3D_redundant_quaternion(q, resolution=100)
        print(fk[-1][1,2])
        #plot_func_3D(forward_kinematics_3D_redundant_quaternion(q, resolution=100))
        plot_func_3D(forward_kinematics_3D_redundant_euler(q, design_paras, resolution=100))


    if forward_kinematics_3D_symbolic:
                # Define symbolic variables
        stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal, len_distal_part_sym, len_enddisk_sym, len_wrist_sym, dist_tendons_sym  = sp.symbols(
            'stem_length alpha robotic_length theta delta_l_niTi rho_proximal rho_distal len_distal_part_sym len_enddisk_sym len_wrist_sym dist_tendons_sym')
        symbol_list = [stem_length, alpha, robotic_length, theta, delta_l_niTi, rho_proximal, rho_distal, len_distal_part_sym, len_enddisk_sym, len_wrist_sym, dist_tendons_sym]
        
        

        values = {
            stem_length: 100,
            alpha: np.deg2rad(40),
            robotic_length: 150,
            theta: np.deg2rad(10),
            delta_l_niTi: 8,
            rho_proximal: np.deg2rad(-60),
            len_distal_part_sym: len_distal_part,
            len_enddisk_sym: len_enddisk,
            len_wrist_sym: len_wrist,
            dist_tendons_sym: dist_tendons,
            rho_distal: sp.atan((-sp.sin(alpha)*sp.cos(rho_proximal)-sp.sin(rho_proximal)*sp.sin(delta_l_niTi/dist_tendons_sym - theta)*sp.cos(alpha))/(sp.cos(alpha)*sp.cos(delta_l_niTi/dist_tendons_sym - theta)))
            # rho_distal : -sp.atan((18000000.0*sp.sin(rho_proximal)*sp.sin(delta_l_niTi/dist_tendons_sym - theta) - sp.sin(rho_proximal)*sp.cos(delta_l_niTi/dist_tendons_sym - theta) 
            #                     + 18000000.0*sp.cos(rho_proximal)*sp.tan(alpha))/(sp.sin(delta_l_niTi/dist_tendons_sym - theta) + 18000000.0*sp.cos(delta_l_niTi/dist_tendons_sym - theta)))
            }

        fk_sym = forward_kinematics_3D_redundant_symbolic(symbol_list, resolution=1)
        T_final = sp.simplify(fk_sym[-1][1,2]).subs(values).subs(values).evalf()
        simplified_result = T_final
        
        print(simplified_result)
        # solved_equ = sp.solve(T_final, rho_distal)
        # print(solved_equ)


    if plot_Ik_3D:
        
        #start configuration
        stem_length = 100
        alpha = 0#30 #in deg, rotation of the robot along the shaft's center point
        robotic_length_travel = 100# 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = 0#30 # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = .1# -15 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = 0 #30 # in deg, wirst joint
        rho_distal = 0 # in deg, camera rotation

        q_initial = [np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]
        fk_inital = forward_kinematics_3D_euler(q_initial)

        psi_y = np.deg2rad(-20) #rot about local y axis
        psi_z = np.deg2rad(30) #rot about local z axis
        psi_x = np.arctan(np.sin(psi_y)*np.tan(psi_z)) # compute orientation about x to keep endoscope pic steady

        X_d = np.array([470, 180, -300, psi_x, psi_y, psi_z]) # Desired pose, [x pos, y pos, z pos, x orien, y orien, z orien] with intrinsic (i.e. local) zyx euler angles ("XYZ" denote extrinsic angles, "xyz" denote intrinsic angles)
        k = 0.05  # Gain
        error_threshold = 1e-4
        
        #q = inverse_kinematics_3D_euler(X_d, k, q_initial, error_threshold)
        q = inverse_kinematics_3D_euler(X_d, k, q_initial, error_threshold) #decrese resolution to 1 in the forward kinematics function for dramatic speed up! 
        print("alpha ", np.rad2deg(q[0]), "in deg")
        print("robotic_length_travel ", q[1], "in mm") #use as is
        print("theta ", np.rad2deg(q[2]), "in deg")
        print("delta_l_niTi ", q[3], "in mm")
        print("rho proximal ", np.rad2deg(q[4]), "in deg")
        print("rho distal ", np.rad2deg(q[5]), "in deg")

        fk = forward_kinematics_3D_euler(q)
        #print("final orientation of the end effector: ", fk[-1][:3,:3])
         
        plot_func_3D(fk, fk_inital, desired_pose=X_d)


    if plot_Ik_3d_redundant:
        
        #start configuration
        stem_length = 50# 100
        alpha = 0 #in deg, rotation of the robot along the shaft's center point
        robotic_length_travel = 100 # 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = 0 #30 # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = 0.001 # -15 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = 0 # in deg, wirst joint
        rho_distal = 0 # in deg, camera rotation
        
        q_initial = [stem_length, np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]
        fk_inital = forward_kinematics_3D_redundant_euler(q_initial, design_paras, resolution=30)

        psi_y = np.deg2rad(2) #rot about local y axis
        psi_z = np.deg2rad(-3) #rot about local z axis
        psi_x = np.arctan(np.sin(psi_y)*np.tan(psi_z)) # compute orientation about x to keep endoscope pic steady

        X_d = np.array([300, 50, 0, psi_x, psi_y, psi_z]) # Desired pose, [x pos, y pos, z pos, x orien, y orien, z orien] with intrinsic (i.e. local) zyx euler angles ("XYZ" denote extrinsic angles, "xyz" denote intrinsic angles)
        k = 0.05  # Gain
        error_threshold = .1
        
        q = inverse_kinematics_3D_redundant_euler(X_d, k, q_initial, error_threshold, design_paras, factor_obj_func=0) #no objective function

        
        # print(f"stem_length          {q[0]:.2f} mm")
        # print(f"alpha               {np.rad2deg(q[1]):.2f} deg")
        # print(f"robotic_length_travel {q[2]:.2f} mm")
        # print(f"theta               {np.rad2deg(q[3]):.2f} deg")
        # print(f"delta_l_niTi        {q[4]:.2f} mm")
        # print(f"rho proximal        {np.rad2deg(q[5]):.2f} deg")
        # print(f"rho distal          {np.rad2deg(q[6]):.2f} deg")

        fk_no_objective_func = forward_kinematics_3D_redundant_euler(q, design_paras, resolution=30)
        
        q = inverse_kinematics_3D_redundant_euler(X_d, k, q_initial, error_threshold, design_paras, factor_obj_func=0.5)

        fk_with_objective_func = forward_kinematics_3D_redundant_euler(q, design_paras, resolution=30)

        plot_func_3D(fk_no_objective_func, fk_with_objective_func, fk_inital, desired_pose=X_d)


    if plot_Ik_3D_redundant_quat:
        
        #start configuration
        stem_length = 156# 100
        alpha = -26# 0 #in deg, rotation of the robot along the shaft's center point
        robotic_length_travel = 28 #100 # 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = 14 #0 #30 # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = -2 # -15 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = -56 #30 # in deg, wirst joint
        rho_distal = -7 # in deg, camera rotation
        
        q_initial = [stem_length, np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]
        fk_inital = forward_kinematics_3D_redundant_quaternion(q_initial, design_paras)

        psi_y = np.deg2rad(-40) #rot about local y axis
        psi_z = np.deg2rad(50) #rot about local z axis
        psi_x = np.arctan(np.sin(psi_y)*np.tan(psi_z)) # compute orientation about x to keep endoscope pic steady

        X_d = np.array([370, 80, 0, psi_x, psi_y, psi_z]) # Desired pose, [x pos, y pos, z pos, x orien, y orien, z orien] with intrinsic (i.e. local) zyx euler angles ("XYZ" denote extrinsic angles, "xyz" denote intrinsic angles)
        k = 0.05  # Gain
        error_threshold = .1
        
        #q = inverse_kinematics_3D_euler(X_d, k, q_initial, error_threshold)
        time_start = time.time()
        q = inverse_kinematics_3D_redundant_quaternion(X_d, k, q_initial, error_threshold, design_paras)
        print("redundant with quaternions took", time.time()-time_start)
        
        print(f"stem_length          {q[0]:.2f} mm")
        print(f"alpha               {np.rad2deg(q[1]):.2f} deg")
        print(f"robotic_length_travel {q[2]:.2f} mm")
        print(f"theta               {np.rad2deg(q[3]):.2f} deg")
        print(f"delta_l_niTi        {q[4]:.2f} mm")
        print(f"rho proximal        {np.rad2deg(q[5]):.2f} deg")
        print(f"rho distal          {np.rad2deg(q[6]):.2f} deg")

        fk = forward_kinematics_3D_redundant_quaternion(q, design_paras)
         
        plot_func_3D(fk, fk_inital, desired_pose=X_d)
        

    if plot_helix_trajectory:
        
        num_poses = 100
        k = 0.05  # Gain
        error_threshold = .1
        ori_limit = 60
        
        #start configuration
        stem_length = 156# 100
        alpha = -26# 0 #in deg, rotation of the robot along the shaft's center point
        robotic_length_travel = 28 #100 # 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = 14 #0 #30 # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = -2 # -15 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = -56 #30 # in deg, wirst joint
        rho_distal = -7 # in deg, camera rotation
        
        q = [stem_length, np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]
        time_list_euler_random = []
        time_list_quat_random = []
        time_list_euler_path = []
        time_list_quat_path = []
        fk_list = []

        # Create a helix path for x, y, z
        t = np.linspace(0, 4 * np.pi, num_poses)  # Parameter for the helix
        x_t_path = 300 + 50 * t  # Helix x-coordinates
        y_t_path = 250 * np.sin(t)  # Helix y-coordinates
        z_t_path = 250 * np.cos(t)  # Helix z-coordinates

        # Gradually change psi_y and psi_z along the path
        psi_y_path = np.linspace(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        psi_z_path = np.linspace(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        psi_x_path = []
        for i in range(num_poses):
            psi_x_path.append(np.arctan(np.sin(psi_y_path[i])*np.tan(psi_z_path[i]))) # compute orientation about x to keep endoscope pic steady

            if i % 30 == 0:
                    X_d = np.array([x_t_path[i], y_t_path[i], z_t_path[i], psi_x_path[i], psi_y_path[i], psi_z_path[i]])  # Desired pose
                    q = inverse_kinematics_3D_redundant_euler(X_d, k, q, error_threshold, design_paras)
                    fk_list.append(forward_kinematics_3D_redundant_euler(q, design_paras, 100))

        plot_helix_path_with_orientation(x_t_path, y_t_path, z_t_path, psi_x_path, psi_y_path, psi_z_path, fk_list)


    if comparison_Ik_3D_redundant_euler_and_quat:
        
        num_poses = 100
        k = 0.05  # Gain
        error_threshold = .1
        ori_limit = 60
        
        #start configuration
        stem_length = 156# 100
        alpha = -26# 0 #in deg, rotation of the robot along the shaft's center point
        robotic_length_travel = 28 #100 # 238 #mm, length of the imaginary circle going through all connections of robotic elements and robotic links
        theta  = 14 #0 #30 # in deg, positive difference will cause rotation in couter clockwise direction
        delta_l_niTi = -2 # -15 # in mm, difference of length of the NiTi wire to the robotic length, positive difference will cause bending in the direction of the trocar's main axis
        rho_proximal = -56 #30 # in deg, wirst joint
        rho_distal = -7 # in deg, camera rotation
        
        q = [stem_length, np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]
        time_list_euler_random = []
        time_list_quat_random = []
        time_list_euler_path = []
        time_list_quat_path = []
        fk_list = []

        # Create a helix path for x, y, z
        t = np.linspace(0, 4 * np.pi, num_poses)  # Parameter for the helix
        x_t_path = 300 + 50 * t  # Helix x-coordinates
        y_t_path = 250 * np.sin(t)  # Helix y-coordinates
        z_t_path = 250 * np.cos(t)  # Helix z-coordinates

        # Gradually change psi_y and psi_z along the path
        psi_y_path = np.linspace(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        psi_z_path = np.linspace(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        psi_x_path = []
        for i in range(num_poses):
            psi_x_path.append(np.arctan(np.sin(psi_y_path[i])*np.tan(psi_z_path[i]))) # compute orientation about x to keep endoscope pic steady

            X_d = np.array([x_t_path[i], y_t_path[i], z_t_path[i], psi_x_path[i], psi_y_path[i], psi_z_path[i]])  # Desired pose
            start_time = time.time()
            inverse_kinematics_3D_redundant_euler(X_d, k, q, error_threshold, design_paras)
            time_list_euler_path.append(time.time()-start_time)
            start_time = time.time()
            q = inverse_kinematics_3D_redundant_quaternion(X_d, k, q, error_threshold, design_paras)
            time_list_quat_path.append(time.time()-start_time)


        x_t = np.random.uniform(300, 600, num_poses)
        y_t = np.random.uniform(-200, 200, num_poses)
        z_t = np.random.uniform(-200, 200, num_poses)
        psi_y = np.random.uniform(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        psi_z = np.random.uniform(np.deg2rad(-ori_limit), np.deg2rad(ori_limit), num_poses)
        q = [stem_length, np.deg2rad(alpha), robotic_length_travel, np.deg2rad(theta), delta_l_niTi, np.deg2rad(rho_proximal), np.deg2rad(rho_distal)]

        for i in range(num_poses):
            psi_x = np.arctan(np.sin(psi_y[i])*np.tan(psi_z[i])) # compute orientation about x to keep endoscope pic steady
            X_d = np.array([x_t[i], y_t[i], z_t[i], psi_x, psi_y[i], psi_z[i]]) # Desired pose, [x pos, y pos, z pos, x orien, y orien, z orien] with intrinsic (i.e. local) zyx euler angles ("XYZ" denote extrinsic angles, "xyz" denote intrinsic angles)

            time_start = time.time()
            inverse_kinematics_3D_redundant_euler(X_d, k, q, error_threshold, design_paras)
            time_list_euler_random.append(time.time()-time_start) 
            time_start = time.time()
            inverse_kinematics_3D_redundant_quaternion(X_d, k, q, error_threshold, design_paras)
            time_list_quat_random.append(time.time()-time_start)
        
        plot_time_boxplot(time_list_euler_random, time_list_quat_random, time_list_euler_path, time_list_quat_path) 

        

