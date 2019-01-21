from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import numpy as np
import rospy

class TLClassifier(object):
    def __init__(self, site):
        
        # look in current path
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        # load model depending on whether running simulated or real data
        if site:
            model = current_dir + '/Trained_Models/real_frozen_inference_graph.pb'
        else:
            model = current_dir + '/Trained_Models/sim_frozen_inference_graph.pb'
  
        self.detection_graph = tf.Graph()
    
        with self.detection_graph.as_default():
    
            od_graph_def = tf.GraphDef()
    
            with tf.gfile.GFile(model, 'rb') as fid:
    
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # initialise model parameters
        self.session = tf.Session(graph=self.detection_graph)       
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        
        image_expanded = np.expand_dims(image, axis=0)
        
        # run trained classifier over image
        (boxes, scores, classes, num) = self.session.run([self.detection_boxes, 
                                                              self.detection_scores, 
                                                              self.detection_classes, 
                                                              self.num_detections],
                                                              feed_dict={self.image_tensor: image_expanded})
    
        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
    
        # identify class with the highest score    
        light_color = TrafficLight.UNKNOWN
        max_score = scores[0]
        max_index = 0
        
        # determine index of most likely class
        for i in range(1, boxes.shape[0]):
            if max_score < scores[i]:
                max_score = scores[i]
                max_index = i
        
        # determine which light state matches with the highest class score
        if classes[max_index] == 1:
            light_color = TrafficLight.GREEN
        elif classes[max_index] == 2:
            light_color = TrafficLight.RED
        elif classes[max_index] == 3:
            light_color = TrafficLight.YELLOW
    
        # if the light colour is ambiguous
        if max_score < 0.5:
            light_color = TrafficLight.UNKNOWN # set light colour as unknown (possibly set as red, safer to stop)

        return light_color        
