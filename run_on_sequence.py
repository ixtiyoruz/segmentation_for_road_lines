import numpy as np
import pickle
import tensorflow as tf
import cv2
import os
from model import ENet_model
from utilities import spatial_softmax
import time 
import math
from utils import transforms
from label_saver import get_shape, save_label, change_pts_size

def one_hot(arr , ncl):
    n, h, w = np.shape(arr)
    tmp = np.zeros((n, h, w, ncl))
    
    for i in range(ncl):
        locs = (arr == i)
        onhot = np.zeros(ncl)
        onhot[i] = 1
        #print(onhot)
        tmp[locs] = onhot
    return tmp
def blend(img1, img2, alpha=0.5):
    beta = 1-alpha
    out = np.uint8(img1* alpha + img2  *beta)
    return out

def getLane_CULane(prob_map, y_px_gap, pts, thresh, resize_shape=[288, 800]):
    """
    Arguments 
    ---------------
    prob_map: prob map for single lane np array size (h, w)
    resize_shape: reshape size target (H, W)
    
    Return:
    -------------------
    coords : x coords buttom up every y_px_gap px, 0 for non exist in resized shape
    """
    
#    Q = 3e-2 # bu bizning estimationimizning xatoligi
#    sz = 11
#    # allocate space for arrays
#    xhat=np.zeros(sz)      # a current estimate of x
#    P=np.zeros(sz)         # a current error estimate
#    xhatminus=np.zeros(sz) # a priori estimate of x
#    Pminus=np.zeros(sz)    # a priori error estimate
#    K=np.zeros(sz)         # gain or blending factor
#    
#    R = 4e-2
#
#    xhat[0] = 0.0
#    P[0] = 1.0

    if(resize_shape is None):
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(pts)
    for i in range(pts):
        
        
        y = int(h - i * y_px_gap / H * h -1)
#        print(y)
        if(y < 0):
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if(line[id] > thresh):
            coords[i] = int(id/ w * W)
#            if(i == 0):
#                xhat[i] = int(id)
#                Pminus[i] = 1.
#                K[i] = 1.
#                P[i] = 1.
#                coords[i] = xhat[i]
#            else:
#                xhatminus[i] = xhat[i-1]
#                Pminus[i] = P[i-1]+Q
#                K[i] = Pminus[i]/( Pminus[i]+R )
#                xhat[i] = xhatminus[i]+K[i]*(int(id)-xhatminus[i])
#                P[i] = (1-K[i])*Pminus[i]
#                coords[i] = xhat[i]
#            print(coords[i], int(id))
    if((coords > 0).sum() < 2):
        coords = np.zeros(pts)
    return coords

def prob2lines_CULane(seg_pred, exist, resize_shape=[288, 800], smooth=True, y_px_gap=20, pts=None, thresh=0.6, no_of_classes=5):
    """
    Arguments:
        -------
        seg_pred: np array size (h, w, 5)
        resize_shape: reshape size target (H, W)
        exist: list of existance, e.g [0, 1, 1, 0]
        smooth: whether to smooth the probability or not
        y_px_gap: y pixel gap for sampling
        pts: how many points for one line
        thresh: probability threshold
    Return:
    ------------
        coordinates [x, y] list of lanes , e.g : [[[9, 568], [50, 549]], [[630, 569],[647, 549]]]
        
    """
    if(resize_shape is None):
        resize_shape = seg_pred.shape[:-1] # seg_pred (h,w,5)
    h, w, _ = seg_pred.shape
    H, W = resize_shape
    coordinates = []
    if(pts is None):
        pts = round(h/2/y_px_gap)
    seg_pred = np.ascontiguousarray(seg_pred)
    for i in range(no_of_classes-1):
        
        prob_map = seg_pred[..., i+1]
        #print(np.shape(prob_map), np.shape(seg_pred))
        if(smooth):
            prob_map = cv2.blur(prob_map, (9,9), borderType=cv2.BORDER_REPLICATE)
        if(exist[i] > 0):
            coords = getLane_CULane(prob_map, y_px_gap, pts, thresh, resize_shape)
            coordinates.append([[coords[j],np.int32( H-1 - j*y_px_gap)] for j in range(pts) if coords[j] > 0])
    return coordinates
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def calculate_distance(focal_length, h_cam, pix_per_meter, h_img):
    Z = focal_length * pix_per_meter * h_cam / h_img
    return Z
def read_camera_parameters():
    with (open("./wide_dist_pickle.p", "rb")) as openfile:
        dist_pickle = pickle.load(openfile)
    mtx = dist_pickle["mtx"]
#    dist = dist_pickle["dist"]
    f1_x = mtx[0][0]
    f1_y = mtx[1][1]
    o_x = mtx[0][2]
    o_y = mtx[1][2]
    pix_per_meter_y = 1.84 / 20#1.19
    pix_per_meter_x = 4*14.8#1.19
    h_cam = 2.5#7.5
    focal_length = math.sqrt(math.pow(f1_x - o_x, 2)  + math.pow(f1_y - o_y, 2))
    res = {'pxmy':pix_per_meter_y,'pxmx':pix_per_meter_x,'focal_length':focal_length,'h_cam':h_cam}
    return res
class Tracking:
    def __init__(self, initial_lane,num_of_features):
        self.initial_state_mean = []
        self.transaction_matrix = []
        for i in range(num_of_features):
            try:
#                print('-')
                self.initial_state_mean.append(initial_lane[i])
            except:
#                print('+')
                self.initial_state_mean.append(initial_lane[-1])
            self.initial_state_mean.append(0)
            
            tt = np.zeros(num_of_features * 2)
            tt1 = np.zeros(num_of_features * 2)
            tt[i * 2] = 1
            tt[i * 2+1] = 1
            tt1[i * 2+1] = 1
            self.transaction_matrix.append(list(tt))
            self.transaction_matrix.append(list(tt1))
#        print(self.transaction_matrix)
#            transition_matrix = [[1, 1, 0, 0],
#                                 [0, 1, 0, 0],
#                                 [0, 0, 1, 1],
#                                 [0, 0, 0, 1]]
#    def update

#def norm_lane(lane, norm_size=3):
#    for i in np.arange(0, len(lane), norm_size):
#        diff = [np.array(lane[i])-np.array(lane[i+1]) for j in range(min(i+norm_size, len(lane) -1))]
#        diff = np.sum(diff, 0) / len(diff)
#        diff = len(np.where(np.array(diff) == 0)[0]) == 0
#        
    
    
def fill_lane(lane, up_row=190, bottom_row=287, limit=2):
    lane_fixed_both = lane.copy()
    if(lane[0][1] < up_row):
        return [],[]
    tmp = lane[0]
    if(len(lane) < limit):
        return np.array(lane), np.array([])
    diff_bottom = [np.array(lane[i])-np.array(lane[i+1]) for i in range(min(limit, len(lane) -1))]
    diff_bottom = np.sum(diff_bottom, 0) / len(diff_bottom)
    diff_b_bottom = len(np.where(np.array(diff_bottom) == 0)[0]) == 0
    
    diff_up = [np.array(lane[i+1])-np.array(lane[i]) for i in np.arange(-1*limit, -1 , 1)]
    diff_up = np.sum(diff_up, 0) / len(diff_up)
    diff_b_up = len(np.where(np.array(diff_up) == 0)[0]) == 0
    
    
    print(diff_up)
    print(diff_bottom)
    if(tmp[1] < bottom_row and diff_b_bottom):
         pr = abs((bottom_row - tmp[1]) / diff_bottom[1])
         tmp = np.array(tmp) + pr * diff_bottom
         lane_fixed_both = [list(tmp)] + lane
         lane = [list(tmp)] + lane
    ## if line points ar not long enough to up_row
    if(lane_fixed_both[-1][1] > up_row and len(lane_fixed_both) >=2 and diff_b_up and up_row > 0):
        
        tmp = lane_fixed_both[-1]
        # how much we need to subtract
        pr = abs((up_row - lane_fixed_both[-1][1]) / diff_up[1])
#        tmp = np.array(tmp) - pr * diff_up
        addition = pr * diff_up
        tmp = [tmp[0] -addition[0],tmp[1] -addition[1]]
        lane_fixed_both =  lane_fixed_both + [list(tmp)]
        lane =  lane + [list(tmp)]

        ## if line points are higher than up_row
    if(lane_fixed_both[-1][1] < up_row and len(lane_fixed_both) >=2 and up_row > 0):
        lane_fixed_both = lane_fixed_both[:-1]
#        print(lane_fixed_both)
#        print(up_row)
        _,lane_fixed_both = fill_lane(lane_fixed_both, up_row=up_row, bottom_row=bottom_row, limit=limit)
    elif(lane_fixed_both[-1][1] < up_row and len(lane_fixed_both) <2 and up_row > 0):
        lane_fixed_both = []
    return np.array(lane), np.array(lane_fixed_both)
if __name__ == '__main__':    
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    project_dir = "./"
    data_dir = project_dir + "data/"
    
    model_id = "model_4x"
    
    batch_size = 1
    img_height = 288
    img_width = 800
    colors = np.array([
                    [0, 0, 0],
                    [255, 0, 0],
                    [ 0, 255, 0],
                    [0, 0, 255],
                    [255,255,0],
                    ],dtype=np.float32)
    no_of_classes = 5   
    model = ENet_model(model_id, img_height=img_height, img_width=img_width,
                batch_size=batch_size, no_classes=no_of_classes)
    train_mean_channels = pickle.load(open("data/mean_channels.pkl", 'rb'))
    input_mean = train_mean_channels#[103.939, 116.779, 123.68] # [0, 0, 0]
    input_std = [1, 1, 1]
    normalizer = transforms.GroupNormalize(mean=(input_mean, (0, )), std=(input_std, (1, )))
    
    # load the mean color channels of the train imgs:
    train_mean_channels = pickle.load(open("data/mean_channels.pkl", 'rb'))
    
    # create a saver for restoring variables/parameters:
    saver = tf.train.Saver(tf.trainable_variables(), write_version=tf.train.SaverDef.V2)
    
    sess =  tf.Session()
    # restore the best trained model:
    saver.restore(sess, "./training_logs/model_augmented_best_epoch_ 12.ckpt")
    
    logits = tf.nn.softmax(model.net.logits, -1)
    exist_logits = model.net.line_existance_logit
    exist_logits = tf.where(tf.less(exist_logits, 0.5), exist_logits * 0.0, exist_logits/exist_logits)
    
    #gaus_kernel = [[]]
    up_row=160
    crop_h = 100
    line_width = 2
    video_name = 'test.mp4'
    cap = cv2.VideoCapture(video_name)
    ret, frm = cap.read()
    counter  = 0
    cam_param = read_camera_parameters()
    
    #means = []
    while(True):
        counter =counter + 1
        if(counter % 3 == 0):
            cap.grab()
            ret, frm = cap.retrieve()
        else:
            cap.grab()
            continue
        original_shape = np.shape(frm)
        imh,imw,_  = original_shape
    #    print(np.shape(frm))
        frame = frm[crop_h:]
        shapes = []
        img = cv2.resize(frame, (img_width, img_height))
        feed_dict = {
            model.imgs_ph : np.expand_dims((img - np.array(input_mean)) / input_std, 0),
            model.early_drop_prob_ph : 0.0,
            model.late_drop_prob_ph :  0.0}
        start = time.time()
        prob,exist = sess.run([logits, exist_logits], feed_dict=feed_dict)
    
    #    print('time usage:', time.time() - start, ' fps:', 1./(time.time() - start))
        lane_coords = prob2lines_CULane(prob[0], exist[0], y_px_gap=15, pts=18,thresh=0.4, resize_shape=None)#np.shape(frm)[:-1])
    #    print(len(lane_coords[0]),len(lane_coords[1]),len(lane_coords[2]),len(lane_coords[3]),)
        colors = [(0,0,0), (0,255,0), (0,255,255), (255,255,0), (255,0,0)]
        direction_pts = [(400,188),(400,288)]
    #    img = cv2.line(img, direction_pts[0], direction_pts[1],[128, 128, 255], 3)
        
        lane1 = lane_coords[0]
        lane2 = lane_coords[1]
        lane3 = lane_coords[2]
        lane4 = lane_coords[3]
        
        
        len1 = len(lane1)
        len2 = len(lane2)
        len3 = len(lane3)
        len4 = len(lane4)
        
        limit_pts = 1
        isFirst_road_exist = (len1>limit_pts) and (len2>limit_pts)
        isSecond_road_exist = (len2>limit_pts) and (len3>limit_pts)
        isThird_road_exist = (len3>limit_pts) and (len4>limit_pts)
        pt_middle_car = direction_pts[1]

	img = cv2.polylines(img, [np.int32(lane1)], False, [128, 128, 255], 3)
	img = cv2.polylines(img, [np.int32(lane2)], False, [128, 128, 255], 3)
	img = cv2.polylines(img, [np.int32(lane3)], False, [128, 128, 255], 3)
	img = cv2.polylines(img, [np.int32(lane4)], False, [128, 128, 255], 3)

        cv2.imshow("img", img)
        k = cv2.waitKey(0)
        if(k  == 27):
            cap.release()
            cv2.destroyAllWindows()
            break

    # validation loss 2 ta epoch oldinga
# 
