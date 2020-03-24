""" 
    author: ikhtiyor
    date: 27.11.2019
    this script is only designed to work with multiple gpus, i think it will work with one gpu as well ))
    original :https://github.com/mcdavid109/Multi-GPU-Training/blob/master/TrainingDemo.ipynb
"""

import tensorflow as tf
from threading import Thread
import cv2
import numpy as np

class Dataset():
    def __init__(self,x_paths, y_paths,y_exist, batch_size, img_height, img_width, no_of_classes=5):
        
        self.x_paths = x_paths
        self.y_paths = y_paths
        self.y_exist = y_exist
        
        self.batch_size = batch_size
        self.len_data = len(x_paths)
        self._make_inputs()
        self.idx = -1
        self.num_threads = 2
        self.num_batch = self.len_data // self.batch_size + 1
        self.img_height = img_height
        self.img_width = img_width
        self.no_of_classes = no_of_classes
        self.augmentators = []
        # this is only for segmentation
        self.layer_idx = np.arange(self.img_height).reshape(self.img_height, 1)
        self.component_idx = np.tile(np.arange(self.img_width), (self.img_height, 1))
    def make_augmentators(self, augment_fc):
        self.augmentators.append(augment_fc)
        
    def _make_inputs(self):
        
        self.inputs = tf.placeholder(shape=[self.img_height,self.img_width,3],dtype=tf.float32,name='data_x')
        self.labels = tf.placeholder(shape=[self.img_height, self.img_width, self.no_of_classes],dtype=tf.int32,name='data_y')
        self.line_existance_labels = tf.placeholder(tf.float32, shape=[self.no_of_classes-1], name="data_existance_y")
        
        self.queue = tf.FIFOQueue(shapes=[[self.img_height,self.img_width,3],[self.img_height, self.img_width, self.no_of_classes], [self.no_of_classes-1]],
                                  dtypes=[tf.float32, tf.float32, tf.float32],
                                  shared_name="fifoqueue",capacity=self.batch_size*2)
        self.enqueue_op = self.queue.enqueue([self.inputs,self.labels, self.line_existance_labels])
        self._queue_close = self.queue.close(cancel_pending_enqueues=True)

    def next_batch(self):
        
        batch_x , batch_y, batch_existance_y = self.queue.dequeue_many(self.batch_size)
        return batch_x, batch_y, batch_existance_y
    


    def close_queue(self, session):

        session.run(self._queue_close)
        
    def _pre_batch_queue(self,sess,coord):
        
        while not coord.should_stop():
            self.idx += 1
            index = self.idx % self.len_data
            
            # read the next img:
            img = cv2.imread(self.x_paths[index], -1)

            # read existance label as well
            train_existance_label= self.y_exist[index]
            
            # read the next label:
            trainId_label = cv2.imread(self.y_paths[index], -1)
            
            for augment_fc in self.augmentators:
                img, trainId_label = augment_fc((img, trainId_label))    

            # convert the label to onehot:
            onehot_label = np.zeros((self.img_height, self.img_width, self.no_of_classes), dtype=np.float32)
            onehot_label[self.layer_idx, self.component_idx, np.int32(trainId_label)] = 1
            
            sess.run(self.enqueue_op,feed_dict = {self.inputs : img,self.labels: onehot_label, self.line_existance_labels:train_existance_label})

    
    def start_queue_threads(self,sess,coord):
        queue_threads = [Thread(target=self._pre_batch_queue, args=(sess, coord))
                         for i in range(self.num_threads)]
        
        for queue_thread in queue_threads:
            coord.register_thread(queue_thread)
            queue_thread.daemon = True
            queue_thread.start()

        return queue_threads
    