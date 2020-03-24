import tensorflow as tf
import os
import pickle
import numpy as np
from utilities import PReLU, spatial_dropout, max_unpool, spatial_softmax,flatten
from utilities import get_iou_loss, distill_loss, l2_loss
from nets.enet import EnetNet
class ENet_model(object):

    def __init__(self, model_id, img_height=288, img_width=800, batch_size=4,no_classes=5):
        self.colors = np.array([
                        [0, 0, 0],  
                        [255, 0, 0],
                        [ 0, 255, 0],
                        [0, 0, 255],
                        [255,255,0],
                        ],dtype=np.float32)
        self.model_id = model_id

        self.project_dir = "./"
        
        self.logs_dir = self.project_dir + "training_logs/"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)

        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.no_of_classes = no_classes
        self.class_weights = pickle.load(open("data/class_weights.pkl", 'rb'))

        self.wd = 1e-4 # (weight decay)
        
        self.lr_const = 1e-3 # (learning rate)
        self.mode ="train"
        # create all dirs for storing checkpoints and other log data:
        self.create_model_dirs()
        
        # add placeholders to the comp. graph:
        self.add_placeholders()

        # define the forward pass, compute logits and add to the comp. graph:
        self.net = EnetNet(self.wd, self.late_drop_prob_ph, self.early_drop_prob_ph, self.no_of_classes)
        self.net.add_logits(self.imgs_ph)

        # compute the batch loss and add to the comp. graph:
        self.loss,self.accuracy, self.recall, self.iou =self.add_loss_op()
        # add a training operation (for minimizing the loss) to the comp. graph:
        self.add_train_op()

    def create_model_dirs(self):
        self.model_dir = self.logs_dir + "model_%s" % self.model_id + "/"
        self.checkpoints_dir = self.model_dir + "checkpoints/"
        self.debug_imgs_dir = self.model_dir + "imgs/"
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.debug_imgs_dir)

    def add_placeholders(self):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, 3],
                    name="ims_ph")
        self.images = tf.image.resize_images(self.imgs_ph, (self.img_width, self.img_width))

        self.onehot_labels_ph = tf.placeholder(tf.float32,
                    shape=[self.batch_size, self.img_height, self.img_width, self.no_of_classes],
                    name="onehot_ls_ph")
        self.line_existance_labels_ph = tf.placeholder(tf.float32, shape=[self.batch_size, self.no_of_classes-1], name="line_exist_label_ph")
        # dropout probability in the early layers of the network:
        self.early_drop_prob_ph = tf.placeholder(tf.float32, name="early_dr_pr_ph")

        # dropout probability in the later layers of the network:
        self.late_drop_prob_ph = tf.placeholder(tf.float32, name="late_dr_pr_ph")
        
        
                      


    def create_feed_dict(self, imgs_batch, early_drop_prob, late_drop_prob, onehot_labels_batch, existance_labels_batch):
        print(np.shape(imgs_batch), np.shape(onehot_labels_batch), np.shape(existance_labels_batch))
        # return a feed_dict mapping the placeholders to the actual input data:
        feed_dict = {}
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.early_drop_prob_ph] = early_drop_prob
        feed_dict[self.late_drop_prob_ph] = late_drop_prob
        feed_dict[self.onehot_labels_ph] = onehot_labels_batch
        feed_dict[self.line_existance_labels_ph] = existance_labels_batch
        return feed_dict
    
 

    def add_loss_op(self):
        
        alpha, beta, gamma = 0.1, 0.1, 0.1
        # compute the weight tensor:
        # in the paper they multiplied the background loss with 0.4, but in my case please decrease it as long as first training step finishes successfully
        # you can know whether training was ok or not, by just checking whether model successfully sepereates four lines, 
        # they do not have to be thin, they just neet to be on their place, so if 0.4 doesnt work start from 0.01 if the learning is too slow then increase it to 0.1
        # if they all fail(in the first epoch prediction will start only outputting background, which is 0), please decrease learning rate and start  again
        # crbg = 0.4
        self.class_weights = [0.01 , 1., 1., 1., 1.] #self.class_weights/(self.class_weights[1])
        weights = self.onehot_labels_ph * self.class_weights
        weights = tf.reduce_sum(weights, 3)

        # compute the weighted cross-entropy segmentation loss for each pixel:
        seg_loss_per_pixel = tf.losses.softmax_cross_entropy(
                    onehot_labels=self.onehot_labels_ph, logits=self.net.logits,
                    weights=weights)
        
        self.line_existance_loss = tf.reduce_mean(tf.squared_difference(self.net.line_existance_logit,self.line_existance_labels_ph ))
        
        ### calculating distill loss
        distill_loss1 = distill_loss(self.net.AT_GEN1, self.net.AT_GEN2)
        distill_loss2 = distill_loss(self.net.AT_GEN2, self.net.AT_GEN3)
        distill_loss3 = distill_loss(self.net.AT_GEN3, self.net.AT_GEN4)
        
        self.distill_loss = tf.reduce_sum([distill_loss1, distill_loss2, distill_loss3])
        
        
        # average the loss over all pixels to get the batch segmentation loss:
        self.seg_loss = tf.reduce_mean(seg_loss_per_pixel)
        accuracy, iou, recall, seg_loss_iou = get_iou_loss(self.onehot_labels_ph, self.net.logits, self.no_of_classes)
        print(np.shape(self.seg_loss), np.shape(seg_loss_iou), np.shape(self.line_existance_loss))
        
        # compute the total loss by summing the segmentation loss and all
        # variable weight decay losses:
        loss = (self.seg_loss + 
                    alpha * seg_loss_iou + 
                    beta * self.line_existance_loss # + gamma * self.distill_loss  # add it after 1 step of training, which is one epoc, in our case 
                    + tf.add_n(tf.get_collection("encoder_wd_losses")) +
                    tf.add_n(tf.get_collection("decoder_wd_losses")) + 
                    tf.add_n(tf.get_collection("line_existance_wd_loss"))
                    )
        # below scripts are only used for summary
        imsum = []
        # Tensorboard inspection
        imsum.append(tf.summary.image('images', self.imgs_ph, max_outputs=1, family=self.mode))
        
        imsum.append(tf.summary.image('GTs', tf.reshape(
            tf.matmul(tf.reshape(self.onehot_labels_ph, [-1, self.no_of_classes]), self.colors), [-1, self.img_height, self.img_width, 3]), max_outputs=1, family=self.mode))
        
        imsum.append(tf.summary.image('responses', tf.reshape(tf.matmul(
            tf.reshape(tf.one_hot(tf.argmax(self.net.logits, -1), self.no_of_classes), [-1, self.no_of_classes]), self.colors),
            [-1, self.img_height, self.img_width, 3]), max_outputs=1, family=self.mode))
        self.image_summary = tf.summary.merge(imsum)
        
        shape = [self.img_height, self.img_width]
        layer1 = tf.image.resize_images( tf.expand_dims(self.net.AT_GEN1*255, -1), shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        layer2 = tf.image.resize_images( tf.expand_dims(self.net.AT_GEN2*255, -1), shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        layer3 = tf.image.resize_images( tf.expand_dims(self.net.AT_GEN3*255, -1), shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        layer4 = tf.image.resize_images( tf.expand_dims(self.net.AT_GEN4*255, -1), shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                
        imatgen = []
        # Tensorboard inspection
        imatgen.append(tf.summary.image('atgen1', layer1, max_outputs=1, family=self.mode+"/atgen"))
        imatgen.append(tf.summary.image('atgen2', layer2, max_outputs=1, family=self.mode+"/atgen"))
        imatgen.append(tf.summary.image('atgen3', layer3, max_outputs=1, family=self.mode+"/atgen"))
        imatgen.append(tf.summary.image('atgen4', layer4, max_outputs=1, family=self.mode+"/atgen"))
        
        self.imatgen_summary = tf.summary.merge(imatgen)
        
        
        
        self.total_loss_sum = tf.get_variable("total_loss", initializer=0.0, dtype=tf.float32) 
        self.total_accuracy_sum = tf.get_variable("total_accuracy", initializer=0.0, dtype=tf.float32)
        self.total_recall_sum = tf.get_variable("total_recall", initializer=0.0, dtype=tf.float32)
        self.total_iou_sum = tf.get_variable("total_iou", initializer=0.0, dtype=tf.float32)
        
        
        
        scalars = []
        loss_scalar = tf.summary.scalar('total_loss', self.total_loss_sum, family=self.mode)
        accuracy_scalar = tf.summary.scalar('total_accuracy', self.total_accuracy_sum, family=self.mode)
        recall_scalar = tf.summary.scalar('total_recall', self.total_recall_sum, family=self.mode)
        iou_scalar = tf.summary.scalar('total_ioul', self.total_iou_sum, family=self.mode)
        
        self.global_step = tf.Variable(0, trainable=False)
        self.rate = tf.train.exponential_decay(self.lr_const, self.global_step, decay_steps = 2*7406, decay_rate = 0.9, staircase=True)
        lr_scalar = tf.summary.scalar('lr', self.rate, family=self.mode)
        scalars.append(loss_scalar)
        scalars.append(accuracy_scalar)
        scalars.append(iou_scalar)
        scalars.append(recall_scalar)
        scalars.append(lr_scalar)
#        scalars.append(learning_rate_scalar)
        
        self.scalars_summary = tf.summary.merge(scalars)
        
        return loss, accuracy, recall, iou
    
    def add_train_op(self):

        # create the train op:
        optimizer = tf.train.AdamOptimizer(self.rate)
        
        self.train_op = optimizer.minimize(self.loss,  global_step=self.global_step)