'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, threshold, device='CPU', extensions=None):
        '''
        instantiating the necessary variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
        self.threshold = threshold
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None


    def load_model(self):
        '''
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.core = IECore()
        self.model = self.core.read_network(self.model_structure, self.model_weights)

        supported_layers = self.core.query_network(network=self.model, device_name=self.device)
        unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]

        if len(unsupported_layers) != 0 and self.device=='CPU':
            log.error("Unsupported layers found: {}".format(unsupported_layers))
            log.error("Check whether extensions are available to add")
            self.core.add_extension(self.extension, self.device)
            supported_layers = self.core.query_network(network=self.model, device_name=self.device)
            unsupported_layers = [l for l in self.model.layers.keys() if l not in supported_layers]
            if len(unsupported_layers) != 0:
                log.error("ERROR: Unsupported layer issue not yet resolved")
                exit(1)
        self.exec_net = self.core.load_network(network=self.model, device_name=self.device, num_requests=1)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape


    def predict(self, image):
        '''
        This method is meant for running predictions on the input image.
        '''
        input_img = self.preprocess_input(image.copy())
        result = self.exec_net.infer({self.input_name:input_img})
        coords = self.preprocess_output(result, image)

        if (len(coords)==0):
            log.error("No face is detected, Next frame will be detected")
            return 0,0

        coord = coords[0]
        cropped_face = image[coord[1]:coord[3], coord[0]:coord[2]]
        
        return coord, cropped_face


    def check_model(self):
        pass


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        #input for this model takes the shape [1x3x384x672] in the format [BxCxHxW]
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame


    def preprocess_output(self, outputs, image):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        # output for this model is [1, 1, N, 7], where N is the number of detected bounding boxes. Each detection has the format [image_id, label, conf, x_min, y_min, x_max, y_max]
        h = image.shape[0]
        w = image.shape[1]
        coords = []
        outs = outputs[self.output_name][0][0]
        for box in outs:
            conf = box[2]
            if conf > self.threshold:
                xmin = int(box[3] * w)
                ymin = int(box[4] * h)
                xmax = int(box[5] * w)
                ymax = int(box[6] * h)
                # if (flagged):
                #     cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255,0,255), 1)
                coords.append([xmin, ymin, xmax, ymax])
        return coords 