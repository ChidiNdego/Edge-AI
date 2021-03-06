import cv2
import os
import numpy as np
import logging as log
import time
from face_detection import FaceDetection
from facial_landmarks_detection import FacialLandmarkDetection
from head_pose_estimation import HeadPoseEstimation
from gaze_estimation import GazeEstimation
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder
import math


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-fd", "--face_detection_model", required=True, type=str, help="Path to face detection model files.")
    parser.add_argument("-fl", "--facial_landmarks_model", required=True, type=str, help="Path to facial landmarks detection model files.")
    parser.add_argument("-hp", "--head_pose_model", required=True, type=str, help="Path to head pose estimation model files.")
    parser.add_argument("-ge", "--gaze_estimation_model", required=True, type=str, help="Path to gaze estimation model files.")
    parser.add_argument("-i", "--input", required=True, type=str, help="Path to image or video file or CAM")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None, help="CPU targeted custom layers.")
    parser.add_argument("-d", "--device", type=str, default="CPU", help="Specify the target device to infer on: CPU, GPU, FPGA or MYRIAD is acceptable. specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5, help="Probability threshold for detections filtering")
    parser.add_argument("-flag", "--visualization_flag", required=False, nargs='+', default=[],
                        help="Example: --flag fd fl hp ge (Seperate each flag by space)"
                             "for see the visualization of different model outputs of each frame,"
                             "fd for Face Detection Model, fl for Facial Landmark Detection Model"
                             "hp for Head Pose Estimation Model, ge for Gaze Estimation Model.")

    return parser


def draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length):
    yaw *= np.pi / 180.0
    pitch *= np.pi / 180.0
    roll *= np.pi / 180.0
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    r_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    r_y = np.array([[math.cos(yaw), 0, -math.sin(yaw)],
                    [0, 1, 0],
                    [math.sin(yaw), 0, math.cos(yaw)]])
    r_z = np.array([[math.cos(roll), -math.sin(roll), 0],
                    [math.sin(roll), math.cos(roll), 0],
                    [0, 0, 1]])

    r = r_z @ r_y @ r_x
    camera_matrix = build_camera_matrix(center_of_face, focal_length)
    xaxis = np.array(([1 * scale, 0, 0]), dtype='float32').reshape(3, 1)
    yaxis = np.array(([0, -1 * scale, 0]), dtype='float32').reshape(3, 1)
    zaxis = np.array(([0, 0, -1 * scale]), dtype='float32').reshape(3, 1)
    zaxis1 = np.array(([0, 0, 1 * scale]), dtype='float32').reshape(3, 1)
    o = np.array(([0, 0, 0]), dtype='float32').reshape(3, 1)
    o[2] = camera_matrix[0][0]
    xaxis = np.dot(r, xaxis) + o
    yaxis = np.dot(r, yaxis) + o
    zaxis = np.dot(r, zaxis) + o
    zaxis1 = np.dot(r, zaxis1) + o
    xp2 = (xaxis[0] / xaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (xaxis[1] / xaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 0, 255), 2)
    xp2 = (yaxis[0] / yaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (yaxis[1] / yaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, (cx, cy), p2, (0, 255, 0), 2)
    xp1 = (zaxis1[0] / zaxis1[2] * camera_matrix[0][0]) + cx
    yp1 = (zaxis1[1] / zaxis1[2] * camera_matrix[1][1]) + cy
    p1 = (int(xp1), int(yp1))
    xp2 = (zaxis[0] / zaxis[2] * camera_matrix[0][0]) + cx
    yp2 = (zaxis[1] / zaxis[2] * camera_matrix[1][1]) + cy
    p2 = (int(xp2), int(yp2))
    cv2.line(frame, p1, p2, (255, 0, 0), 2)
    cv2.circle(frame, p2, 3, (255, 0, 255), 2)
    
    return frame


def build_camera_matrix(center_of_face, focal_length):
    cx = int(center_of_face[0])
    cy = int(center_of_face[1])
    camera_matrix = np.zeros((3, 3), dtype='float32')
    camera_matrix[0][0] = focal_length
    camera_matrix[0][2] = cx
    camera_matrix[1][1] = focal_length
    camera_matrix[1][2] = cy
    camera_matrix[2][2] = 1
    
    return camera_matrix


def main():
    # command line args
    args = build_argparser().parse_args()
    input_file_path = args.input
    log_object = log.getLogger()
    oneneneflags = args.visualization_flag

    # Initialise the classes
    fd_object = FaceDetection(model_name=args.face_detection_model, device=args.device, threshold=args.prob_threshold, extensions=args.cpu_extension)
    fl_object = FacialLandmarkDetection(model_name=args.facial_landmarks_model, device=args.device, extensions=args.cpu_extension)
    hp_object = HeadPoseEstimation(model_name=args.head_pose_model, device=args.device, extensions=args.cpu_extension)
    ge_object = GazeEstimation(model_name=args.gaze_estimation_model, device=args.device, extensions=args.cpu_extension)
    
    mouse_controller_object = MouseController('low', 'fast')
    
    ### Loading the models ###
    log_object.error("=================== Models Load Time ====================")
    start_time = time.time()
    fd_object.load_model()
    log_object.error("Face detection model loaded in {:.3f} ms".format((time.time() - start_time) * 1000))
    
    fl_start = time.time()
    fl_object.load_model()
    log_object.error("Facial landmarks detection model loaded in {:.3f} ms".format((time.time() - fl_start) * 1000))
    
    hp_start = time.time()
    hp_object.load_model()
    log_object.error("Head pose estimation model loaded in {:.3f} ms".format((time.time() - hp_start) * 1000))
    
    ge_start = time.time()
    ge_object.load_model()
    log_object.error("Gaze estimation model loaded in {:.3f} ms".format((time.time() - ge_start) * 1000))
    
    total_time = time.time() - start_time
    log_object.error("=================== Models loaded successfully ===================")
    log_object.error("Total loading time is {:.3f} ms".format(total_time * 1000))

    counter = 0
    infer_start = time.time()
    log_object.error("=================== Start inferencing on input video ====================")
    
    if input_file_path == "CAM":
        input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)
        
        log_object.error("Input feeders are loaded")
        input_feeder.load_data()

    for frame in input_feeder.next_batch():
        # if not flag:
        #     break
        pressed_key = cv2.waitKey(60)
        counter += 1

        face_coordinates, face_image = fd_object.predict(frame.copy())
        if face_coordinates == 0:
            continue
        
        hp_output = hp_object.predict(face_image)

        left_eye_image, right_eye_image, eye_coord = fl_object.predict(face_image)

        mouse_coordinate, gaze_vector = ge_object.predict(left_eye_image, right_eye_image, hp_output)

        if len(oneneneflags) != 0:
            preview_window = frame.copy()
            if 'fd' in oneneneflags:
                if len(oneneneflags) != 1:
                    preview_window = face_image
                else:
                    cv2.rectangle(preview_window, (face_coordinates[0], face_coordinates[1]), (face_coordinates[2], face_coordinates[3]), (0, 150, 0), 3)
            if 'fl' in oneneneflags:
                if not 'fd' in oneneneflags:
                    preview_window = face_image.copy()
                cv2.rectangle(preview_window, (eye_coord[0][0], eye_coord[0][1]), (eye_coord[0][2], eye_coord[0][3]), (150, 0, 150))
                cv2.rectangle(preview_window, (eye_coord[1][0], eye_coord[1][1]), (eye_coord[1][2], eye_coord[1][3]), (150, 0, 150))
            if 'hp' in oneneneflags:
                cv2.putText(preview_window, "yaw:{:.1f} | pitch:{:.1f} | roll:{:.1f}".format(hp_output[0], hp_output[1], hp_output[2]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            if 'ge' in oneneneflags:

                yaw = hp_output[0]
                pitch = hp_output[1]
                roll = hp_output[2]
                focal_length = 950.0
                scale = 50
                center_of_face = (face_image.shape[1] / 2, face_image.shape[0] / 2, 0)
                if 'fd' in oneneneflags or 'fl' in oneneneflags:
                    draw_axes(preview_window, center_of_face, yaw, pitch, roll, scale, focal_length)
                else:
                    draw_axes(frame, center_of_face, yaw, pitch, roll, scale, focal_length)

        if len(oneneneflags) != 0:
            img_hor = np.hstack((cv2.resize(frame, (500, 500)), cv2.resize(preview_window, (500, 500))))
        else:
            img_hor = cv2.resize(frame, (500, 500))

        cv2.imshow('Visualization', img_hor)
        mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])

        if pressed_key == 27:
            log_object.error("exit key is pressed..")
            break
    
    infer_time = round(time.time() - infer_start, 1)
    fps = int(counter) / infer_time
    log_object.error("counter {} seconds".format(counter))
    log_object.error("total inference time {} seconds".format(infer_time))
    log_object.error("fps {} frame/second".format(fps))
    log_object.error("Video session has ended")
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stats.txt'), 'w') as f:
        f.write(str(infer_time) + '\n')
        f.write(str(fps) + '\n')
        f.write(str(total_time) + '\n')

    input_feeder.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()