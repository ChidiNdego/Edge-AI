"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image, webcam, or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.2,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-c", "--box_color", required=False, type=str,                                       default='MAGENTA', help="Color of bounding box")
    parser.add_argument("-pc", "--perf_counts", type=str, default=False,
                        help="Print performance counters")    
    return parser


def performance_counts(perf_count):
    """
    Print information about layers of the model.
    :param perf_count: Dictionary consists of status of the layers.
    :return: None
    """
    print("{:<70} {:<15} {:<15} {:<15} {:<10}".format('name', 'layer_type',
                                                      'exec_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_count.items():
        print("{:<70} {:<15} {:<15} {:<15} {:<10}".format(layer,
                                                          stats['layer_type'],
                                                          stats['exec_type'],
                                                          stats['status'],
                                                          stats['real_time']))
        
def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Golden Yellow if an invalid color is given.
    '''
    colors={"BLUE":(255,0,0), "GREEN":(0,255,0), "RED":(0,0,255), "YELLOW":(0,255,255), "MAGENTA":(204,0,204)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['MAGENTA']

def draw_boxes(frame, result, args, width, height):
    '''
    Draw bounding box on detected object.
    Tracks count of bounding box drawn
    '''
    pointer = 0
    probs = result[0, 0, :, 2]
    for i, p in enumerate(probs):
        if p > args.pt:
            pointer += 1
            box = result[0, 0, i, 3:]
            p1 = (int(box[0] * width), int(box[1] * height))
            p2 = (int(box[2] * width), int(box[3] * height))
            frame = cv2.rectangle(frame, p1, p2, args.c, 3)
    return frame, pointer

def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    #Rename arguments for simplicity
    args.m = args.model
    args.l = args.cpu_extension
    args.d = args.device
    args.i = args.input
    args.pc = args.perf_counts
    args.pt = args.prob_threshold
    
    # Initialise the class
    infer_network = Network()
    
    ###Set color for bounding box
    args.c = convert_color(args.box_color)
    
    #default request number
    num_requests=0
    
    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.m, num_requests, args.d, CPU_EXTENSION)

    ### TODO: Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()

    # Flag for the input image
    single_image_mode = False
    
    # Checks for input image
    if args.i.endswith('.jpg') or args.i.endswith('.bmp') :
        single_image_mode = True
        input_stream = args.i
        
    # Checks for live feed
    elif args.i == 'CAM':
        input_stream = 0

    # Checks for video file
    else:
        input_stream = args.i

    cap = cv2.VideoCapture(input_stream)
    
    ### Get and open video
    cap.open(args.i)
    
    ###Grab the shape of the input
    width = int(cap.get(3))
    height = int(cap.get(4))
    
    #print(net_input_shape)
    input_shape = net_input_shape['image_tensor']

    #assign initial values for publish message values
    prev_dur = 0
    counter_total = 0
    dur = 0
    request_id=0 
    report = 0
    counter = 0
    prev_counter = 0

    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break
        
        ###Exits video when esc button is pressed ###
        key_pressed = cv2.waitKey(60)
        if key_pressed == 27:
            break
        
        ### TODO: Pre-process the image as needed ###
        #p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = cv2.resize(frame, (input_shape[3], input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        ### TODO: Start asynchronous inference for specified request ###
        net_input = {'image_tensor': p_frame,'image_info': p_frame.shape[1:]}
        duration_report = None
        infer_network.exec_net(net_input, req_id=request_id)
        #infer_network.exec_net(p_frame, request_id)
        
        infer_start = time.time()
        
        ### TODO: Wait for the result ###
        if infer_network.wait(request_id) == 0:
            det_time = time.time() - infer_start
            result = infer_network.get_output(request_id)
            ### TODO: Extract any desired stats from the results ###        
            
            frame, pointer = draw_boxes(frame, result, args, width, height)
            
            inf_time_message = "Inference time: {:.3f}ms"\
                               .format(det_time * 1000)
            cv2.putText(frame, inf_time_message, (15, 20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, args.c, 1)

        
            if pointer != counter: #if it identifies a person
                #start_time = time.time()
                prev_counter = counter
                counter = pointer
                if dur >= 3:
                    prev_dur = dur
                    dur = 0
                else:
                    dur = prev_dur + dur
                    prev_dur = 0  # unknown, not needed in this case
            else:
                dur += 1
                if dur >= 3:
                    report = counter
                    if dur == 3 and counter > prev_counter:
                        counter_total += counter - prev_counter
                    elif dur == 3 and counter < prev_counter:
                        duration_report = int(prev_dur / 10.0)
                        #duration = int(time.time() - start_time)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person', payload=json.dumps({'count': report, 
                                                         'total': counter_total}))
            cv2.putText(frame, "{} Count(s) in view".format(str(report)), (15, 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, convert_color("GREEN"), 1)
            cv2.putText(frame, "Total Count: "+str(counter_total), (15, 60), cv2.FONT_HERSHEY_COMPLEX, 0.5, convert_color("GREEN"), 1)

            if duration_report:
                client.publish("person/duration",
                               json.dumps({"duration": duration_report}))
            #cv2.putText(frame, "Duration: "+str(duration_report), (15, 80), cv2.FONT_HERSHEY_COMPLEX, 0.5, convert_color("GREEN"), 1)
            
        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        ### TODO: Write an output image if `single_image_mode` ###
        if single_image_mode:
            cv2.imwrite('output_image.jpg', frame)

    # Release the out writer, capture, and destroy any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()