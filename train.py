import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import random
import os
from utils import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
from utilities import label_img_to_color
from model import ENet_model
project_dir = "./"
data_dir = project_dir + "data/"

# change this to not overwrite all log data when you train the model:
model_id = "model_4x_only_lane"

batch_size = 12
img_height = 288
img_width = 800

no_of_epochs = 100

class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs

def evaluate_on_val():
    #    random.shuffle(val_data)
    #    val_img_paths, val_trainId_label_paths = zip(*val_data)
    
    val_batch_losses = []
    batch_pointer = 0
    for step in range(no_of_val_batches):
        batch_imgs = np.zeros((batch_size, img_height, img_width, 1), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)
        batch_existance_labels = np.zeros((batch_size, 4), dtype=np.float32)

        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(val_img_paths[batch_pointer + i], -1)
            
            img = cv2.resize(img, ( img_width,img_height ),interpolation = cv2.INTER_NEAREST)
            
            # read the next label:
            trainId_label = cv2.imread(train_trainId_label_paths[batch_pointer + i], -1)
            if(np.random.randint(0,2,1) == 0):
                img = img[100:, :, :]
                trainId_label = trainId_label[100:, :]
            
            if(np.random.randint(0,2,1) == 0):
                img, trainId_label = scaler((img, trainId_label))
                img, trainId_label = cropper((img, trainId_label))
    
            trainId_label = cv2.resize(trainId_label, ( img_width,img_height ),interpolation = cv2.INTER_NEAREST)
            img, trainId_label = normalizer((img, trainId_label))
            batch_imgs[i] = img

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, np.int32(trainId_label)] = 1
            batch_onehot_labels[i] = onehot_label
            
#            train_existance_label = np.array([0, 0, 0, 0])
#            train_existance_label[np.unique(trainId_label)[np.unique(trainId_label)>0] - 1] = 1
            # read existance label as well
            train_existance_label= val_existance_labels[batch_pointer + i]
            batch_existance_labels[i] = train_existance_label
            
        batch_pointer += batch_size

        batch_feed_dict = model.create_feed_dict(imgs_batch=batch_imgs,
                    early_drop_prob=0.0, late_drop_prob=0.0,
                    onehot_labels_batch=batch_onehot_labels, existance_labels_batch = batch_existance_labels)
        
        # run a forward pass, get the batch loss and the logits:
        batch_loss, logits = sess.run([model.loss, model.net.logits],
                    feed_dict=batch_feed_dict)

        val_batch_losses.append(batch_loss)
        print ("epoch: %d/%d, val step: %d/%d, val batch loss: %g" % (epoch+1,
                    no_of_epochs, step+1, no_of_val_batches, batch_loss))

        if step < 4:
            # save the predicted label images to disk for debugging and
            # qualitative evaluation:
            predictions = np.argmax(logits, axis=3)
            for i in range(batch_size):
                pred_img = predictions[i]
                label_img_color = label_img_to_color(pred_img)
                cv2.imwrite((model.debug_imgs_dir + "val_" + str(epoch) + "_" +
                            str(step) + "_" + str(i) + ".png"), label_img_color)

    val_loss = np.mean(val_batch_losses)
    return val_loss

train_mean_channels = pickle.load(open("data/mean_channels.pkl", 'rb'))
input_mean = train_mean_channels#[103.939, 116.779, 123.68] # [0, 0, 0]
input_std = [1, 1, 1]
ignore_label = 255
scaler = transforms.GroupRandomScale(size=(0.595, 0.621), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST))
cropper = transforms.GroupRandomCropRatio(size=(img_width, img_height))
rotater= transforms.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(input_mean, (0, )))
normalizer = transforms.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, )))

def train_data_iterator():
    global train_img_paths, train_trainId_label_paths, train_existance_labels
    #shuffling the train data
    indexes = list(range(len(train_img_paths)))
    random.shuffle(indexes)
    
    train_img_paths = train_img_paths[indexes]
    train_trainId_label_paths = train_trainId_label_paths[indexes]
    train_existance_labels = train_existance_labels[indexes]
   
    batch_pointer = 0
    for step in range(no_of_batches):
        # get and yield the next batch_size imgs and onehot labels from the train data:
        batch_imgs = np.zeros((batch_size, img_height, img_width, 1), dtype=np.float32)
        batch_onehot_labels = np.zeros((batch_size, img_height, img_width,
                    no_of_classes), dtype=np.float32)
        batch_existance_labels = np.zeros((batch_size, 4), dtype=np.float32)
        for i in range(batch_size):
            # read the next img:
            img = cv2.imread(train_img_paths[batch_pointer + i], -1)
            print(train_img_paths[batch_pointer + i])
            img = cv2.resize(img, ( img_width,img_height ),interpolation = cv2.INTER_NEAREST)
            #img = img - train_mean_channels
            
            # read the next label:
            trainId_label = cv2.imread(train_trainId_label_paths[batch_pointer + i], -1)
            trainId_label = cv2.resize(trainId_label, ( img_width,img_height ),interpolation = cv2.INTER_NEAREST)
            
            # this is image augmnetation level change it if you want
            if(np.random.randint(0,2,1) == 0):
                img = img[100:, :, :]
                trainId_label = trainId_label[100:, :]
            if(np.random.randint(0,2,1) == 0):
                img, trainId_label = scaler((img, trainId_label))
            img, trainId_label = cropper((img, trainId_label))
            img, trainId_label = rotater((img, trainId_label))
            img, trainId_label = normalizer((img, trainId_label))
            
            batch_imgs[i] = img

            # convert the label to onehot:
            onehot_label = np.zeros((img_height, img_width, no_of_classes), dtype=np.float32)
            onehot_label[layer_idx, component_idx, np.int32(trainId_label)] = 1
            #print(np.shape(batch_existance_labels))
            batch_onehot_labels[i] = onehot_label
            
             # read existance label as well
            train_existance_label= train_existance_labels[batch_pointer + i]
            
#            train_existance_label = np.array([0, 0, 0, 0])
#            train_existance_label[np.unique(trainId_label)[np.unique(trainId_label)>0] - 1] = 1
            batch_existance_labels[i] = train_existance_label
            
        batch_pointer += batch_size

        yield (batch_imgs, batch_onehot_labels, batch_existance_labels)    

def load(sess, saver, checkpoint_dir, model_id):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, model_id)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            #counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            ##return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False
            
with tf.Session() as sess:
    is_beginning = True
    # load the training data from disk:
    train_img_paths = np.array(pickle.load(open(data_dir + "train_img_paths.pkl", 'rb')))
    train_trainId_label_paths = np.array(pickle.load(open(data_dir + "train_trainId_label_paths.pkl", 'rb')))
    train_existance_labels = np.array(pickle.load(open(data_dir + "train_existance_label.pkl", 'rb')))
    #train_data = list(zip(train_img_paths, train_trainId_label_paths))

    #shuffling the train data
    indexes = list(range(len(train_img_paths)))
    random.shuffle(indexes)
    
    train_img_paths = train_img_paths[indexes]
    train_trainId_label_paths = train_trainId_label_paths[indexes]
    train_existance_labels = train_existance_labels[indexes]
    
    # compute the number of batches needed to iterate through the training data:
    no_of_train_imgs = len(train_img_paths)
    no_of_batches = int(no_of_train_imgs/batch_size)

    # load the validation data from disk:
    val_img_paths = pickle.load(open(data_dir + "val_img_paths.pkl", 'rb'))
    val_trainId_label_paths = pickle.load(open(data_dir + "val_trainId_label_paths.pkl", 'rb'))
    val_existance_labels = pickle.load(open(data_dir + "val_existance_label.pkl", 'rb'))
    #val_data = zip(val_img_paths, val_trainId_label_paths)

    # compute the number of batches needed to iterate through the val data:
    no_of_val_imgs = len(val_img_paths)
    no_of_val_batches = int(no_of_val_imgs/batch_size)

    # define params needed for label to onehot label conversion:
    layer_idx = np.arange(img_height).reshape(img_height, 1)
    component_idx = np.tile(np.arange(img_width), (img_height, 1))




    model = ENet_model(model_id, img_height=img_height, img_width=img_width,
                batch_size=batch_size)

    no_of_classes = model.no_of_classes
    # load the mean color channels of the train imgs:

    # create a saver for saving all model variables/parameters:
    saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2, max_to_keep = 100000)

    # initialize all log data containers:
    train_loss_per_epoch = []
    val_loss_per_epoch = []

    # initialize a list containing the 5 best val losses (is used to tell when to
    # save a model checkpoint):
    best_epoch_losses = [1000, 1000, 1000, 1000, 1000]
    train_dir = "./logs/model" + model.model_id
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    summary_writer = tf.summary.FileWriter(train_dir)
    summary_op = tf.summary.merge_all()
    
    
    
    # initialize all variables/parameters:
    init = tf.global_variables_initializer()
    sess.run(init)
    could_load = load(sess, saver, model.logs_dir,"model_%s" % model.model_id + "/checkpoints/")
    global_step_counter = 0
    for epoch in range(no_of_epochs):
        print( "###########################")
        print( "######## NEW EPOCH ########")
        print( "###########################")
        print( "epoch: %d/%d" % (epoch+1, no_of_epochs))
        
        # run an epoch and get all batch losses:
        batch_losses = []
        batch_accuracies = []
        batch_recalls = []
        batch_ious = []
        for step, (imgs, onehot_labels, batch_existance_labels) in enumerate(train_data_iterator()):
            global_step_counter +=1 
            feed_dict = {
            model.imgs_ph : imgs,
            model.early_drop_prob_ph : 0.1,
            model.late_drop_prob_ph : 0.3,
            model.onehot_labels_ph : onehot_labels,
            model.line_existance_labels_ph : batch_existance_labels}

            # compute the batch loss and compute & apply all gradients w.r.t to
            # the batch loss (without model.train_op in the call, the network
            # would NOT train, we would only compute the batch loss):
            if(step % 50 == 0):
                

                batch_loss, accuracy, recall,iou, _ , summary_str1,  summary_str2, summary_str3= sess.run([model.loss,
                                         model.accuracy,model.recall,model.iou, 
                                         model.train_op, model.image_summary, model.imatgen_summary,
                                         model.scalars_summary],feed_dict=feed_dict)
                batch_losses.append(batch_loss)
                batch_accuracies.append(accuracy)
                batch_recalls.append(recall)
                batch_ious.append(iou)                         
                
                total_loss = np.mean(batch_losses)
                total_accuracy = np.mean(batch_accuracies)
                total_recall = np.mean(batch_recalls)
                total_iou = np.mean(batch_ious)

                assign_temp1 =model.total_loss_sum.assign(total_loss)
                assign_temp2 =model.total_accuracy_sum.assign(total_accuracy)
                assign_temp3 =model.total_recall_sum.assign(total_recall)
                assign_temp4 =model.total_iou_sum.assign(total_iou)
                sess.run([assign_temp1,assign_temp2, assign_temp3, assign_temp4])

                
                summary_writer.add_summary(summary_str1, global_step = int(global_step_counter/50))
                summary_writer.add_summary(summary_str2, global_step = int(global_step_counter/50))
                summary_writer.add_summary(summary_str3, global_step = int(global_step_counter/50))
            else:
                batch_loss, accuracy, recall,iou, _ = sess.run([model.loss,model.accuracy,
                model.recall,model.iou, model.train_op],
                            feed_dict=feed_dict)
                batch_losses.append(batch_loss)
                batch_accuracies.append(accuracy)
                batch_recalls.append(recall)
                batch_ious.append(iou) 
            
            
            
            print( "step: %d/%d, training batch loss: %g" % (step+1, no_of_batches, batch_loss))
            
            if(step > 0 and step % 500 == 0):
                # save the model weights to disk:
                checkpoint_path = (model.checkpoints_dir + "model_" +
                            model.model_id + "_step_" + str(step) + "_epoch_"+ str(epoch)  +".ckpt")
                saver.save(sess, checkpoint_path)
                print( "checkpoint saved in file: %s" % checkpoint_path)
#            if(epoch == 0):break;
                    
        # compute the train epoch loss:
        train_epoch_loss = np.mean(batch_losses)
        # save the train epoch loss:
        train_loss_per_epoch.append(train_epoch_loss)
        if(is_beginning):
            try:
                train_loss_saved = pickle.load(open("%strain_loss_per_epoch.pkl"% model.model_dir, 'rb'))
#                if(np.any(train_loss_saved)):
#                    train_loss_saved = list(train_loss_saved[np.argwhere(np.logical_not(np.isnan(train_loss_saved) ))])
#                else:
#                    train_loss_saved = list(train_loss_saved)
                train_loss_saved.extend(train_loss_per_epoch)
                train_loss_per_epoch = train_loss_saved
            except OSError as e:
                print(str(e), 'this is big begining, with zero background')
        # save the train epoch losses to disk:
        pickle.dump(train_loss_per_epoch, open("%strain_loss_per_epoch.pkl"
                    % model.model_dir, "wb"))
        print( "training loss: %g" % train_epoch_loss)

        # run the model on the validation data:
        val_loss = evaluate_on_val()
        # save the val epoch loss:
        val_loss_per_epoch.append(val_loss)
        
        if(is_beginning):
            try:
                val_loss_saved = pickle.load(open("%sval_loss_per_epoch.pkl"% model.model_dir, 'rb'))
                val_loss_saved.extend(val_loss_per_epoch)
                val_loss_per_epoch = val_loss_saved
                is_beginning = False
            except OSError as e:
                    print(str(e), 'this is big begining, with zero background')
        # save the val epoch losses to disk:
        pickle.dump(val_loss_per_epoch, open("%sval_loss_per_epoch.pkl"\
                    % model.model_dir, "wb"))
        print( "validation loss: %g" % val_loss)

        if val_loss < max(best_epoch_losses): # (if top 5 performance on val:)
            # save the model weights to disk:
            checkpoint_path = (model.checkpoints_dir + "model_" +
                        model.model_id + "_best_epoch_ "+ str(epoch)  + ".ckpt")
            saver.save(sess, checkpoint_path)
            print( "checkpoint saved in file: %s" % checkpoint_path)

            # update the top 5 val losses:
            index = best_epoch_losses.index(max(best_epoch_losses))
            best_epoch_losses[index] = val_loss
        
        # plot the training loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(train_loss_per_epoch, "k^")
        plt.plot(train_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("training loss per epoch")
        plt.savefig("%strain_loss_per_epoch.png" % model.model_dir)
        plt.close(1)

        # plot the val loss vs epoch and save to disk:
        plt.figure(1)
        plt.plot(val_loss_per_epoch, "k^")
        plt.plot(val_loss_per_epoch, "k")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.title("validation loss per epoch")
        plt.savefig("%sval_loss_per_epoch.png" % model.model_dir)
        plt.close(1)