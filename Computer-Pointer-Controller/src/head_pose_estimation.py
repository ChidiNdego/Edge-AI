'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import numpy as np
import cv2
import logging as log
from openvino.inference_engine import IENetwork, IECore

class HeadPoseEstimation:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        instantiating the necessary variables.
        '''
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.extensions = extensions
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
        outputList = self.preprocess_output(result)

        return outputList


    def check_model(self):
        pass


    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        output = []
        output.append(outputs['angle_y_fc'].tolist()[0][0])
        output.append(outputs['angle_p_fc'].tolist()[0][0])
        output.append(outputs['angle_r_fc'].tolist()[0][0])

        return output
