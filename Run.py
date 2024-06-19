"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


from torch_lib.Dataset import *
from library.Math import *
from library.Plotting import *
from torch_lib import Model, ClassAverages
from yolo.yolo import cv_Yolo
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
import repvgg_pytorch as repvgg
import argparse
#from TRT import TRT
#===Use ONNX to speed up===
import onnx
import onnxruntime as ort
import sys
from PyQt5.QtWidgets import QApplication
from CAR_interface import Ui_CarAccidentRecognition
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()

parser.add_argument("--image-dir", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                    is eval/image_2/")

# TODO: support multiple cal matrix input types
parser.add_argument("--cal-dir", default="camera_cal/",
                    help="Relative path to the directory containing camera calibration form KITTI. \
                    Default is camera_cal/")

parser.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                    By default, this will pull images from ./eval/video")

parser.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

parser.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")


def plot_regressed_3d_bbox(img, bev_img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):

    # the math! returns X, the corners used for constraint
    location, X = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)
    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes
    plot_bev(bev_img, location, orient, dimensions, scale=10, image_size=(500, 500))
    return location

def main():
    '''=============ADD THE UI PART=============='''
    app = QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_CarAccidentRecognition()
    ui.setupUi(MainWindow)
    MainWindow.show()
    FLAGS = parser.parse_args()
    '''=========================================='''
    # load torch
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    onnx_path = weights_path + "/model.onnx"
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.engine')]
    onnx_model = onnx.load(onnx_path)
    ort_session = ort.InferenceSession(onnx_path)

    #model_path = './weights/epoch_70.trt'
    #trt_model = TRT(model_path=model_path, fp16=True)
    #trt_model.start()
    '''
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        #my_efficeintnet = EfficientNet.from_pretrained('efficientnet-b0')
        # TODO: load bins from file or something
        model = Model.Model(model_name='efficientnet-b0').cuda()
        #model = Model.Model(model_name='RepVGG-A0', deploy = True, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    '''
    # load yolo
    yolo_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/best.pt'

    yolo = cv_Yolo(yolo_path)

    averages = ClassAverages.ClassAverages()

    # TODO: clean up how this is done. flag?
    angle_bins = generate_bins(8)

    image_dir = FLAGS.image_dir
    cal_dir = FLAGS.cal_dir
    if FLAGS.video:
        if FLAGS.image_dir == "eval/image_2/" and FLAGS.cal_dir == "camera_cal/":
            image_dir = "eval/video/2011_09_26/image_2/"
            cal_dir = "eval/video/2011_09_26/"


    #img_path = os.path.abspath(os.path.dirname(__file__)) + "/" + image_dir
    img_path = 'C:/Users/99/Desktop/GitHub/3D-BoundingBox/eval/image_2/'
    #img_path = 'C:/Users/99/Desktop/GitHub/3D-BoundingBox/eval/image_horizon4/'
    # using P_rect from global calibration file
    #calib_path = os.path.abspath(os.path.dirname(__file__)) + "/" + cal_dir
    #calib_file = calib_path + "calib_cam_to_cam.txt"
    calib_file = 'C:/Users/99/Desktop/GitHub/3D-BoundingBox/eval/calib/calib_cam_to_cam.txt'
    calib_path = 'C:/Users/99/Desktop/GitHub/3D-BoundingBox/Kitti/training/calib'

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except:
        print("\nError: no images in %s"%img_path)
        exit()

    for img_id in ids:

        start_time = time.time()

        img_file = img_path + img_id + ".png"

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        truth_img =  cv2.cvtColor(truth_img, cv2.COLOR_BGR2RGB)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        start_time_yolo = time.time()
        detections = yolo.detect(yolo_img)
        end_time_yolo = time.time()
        print("yolo prediction time = ", end_time_yolo - start_time_yolo,"seconds")
        bev_img = np.zeros((500, 500, 3), dtype=np.uint8) + 255
        for detection in detections:
            if not averages.recognized_class(detection.detected_class):
                continue
            #check if detected object is too snall or not, 
            if (detection.box_2d[1][0] - detection.box_2d[0][0]) * (detection.box_2d[1][1] - detection.box_2d[0][1]) < 5000:
                continue
            # this is throwing when the 2d bbox is invalid
            # TODO: better check
            try:
                detectedObject = DetectedObject(img, detection.detected_class, detection.box_2d, calib_file)
            except:
                continue

            theta_ray = detectedObject.theta_ray
            theta_ray = 0
            input_img = detectedObject.img
            input_img_np = input_img.permute(1, 2, 0).cpu().detach().numpy()
            #.imshow(input_img_np)
            #plt.show()
            input_img_np = cv2.cvtColor(input_img_np, cv2.COLOR_BGR2RGB)
            #plt.imshow(input_img_np)
            #plt.show()
            proj_matrix = detectedObject.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class
            input_tensor = torch.zeros([1,3,224,224]).to(device) #1, 3, 224, 224
            input_tensor[0,:,:,:] = input_img
            input_data = {"input.1": input_tensor.cpu().numpy()}
            start_time_eff = time.time()
            output_ =  ort_session.run(None, input_data)
            #conf_ = output_
            #print(f"orient_:{orient_}")
            #output = trt_model.predict(input_tensor)
            end_time_eff = time.time()
            #print("model prediction time = ", end_time_eff - start_time_eff,"seconds")
            [conf_] = output_

            #conf_ = conf_.cpu().numpy()
            #print(f"conf:{conf_}")


            
            dim = averages.get_item(detected_class)

            argmax = np.argmax(conf_)
            alpha = (5 * argmax - 90.) * np.pi / 180.0
            if(alpha >= 1.5*np.pi):
                alpha -= 2* np.pi
            print(f"argmax: {argmax}")
            print(f"alpha: {alpha}")
            if FLAGS.show_yolo:
                location = plot_regressed_3d_bbox(img, bev_img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(img, bev_img, proj_matrix, box_2d, dim, alpha, theta_ray)


        if FLAGS.show_yolo:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('SPACE for next image, any other key to exit', numpy_vertical)
            cv2.imshow('BEV', bev_img)
        else:   
            ui.update_bev_image(bev_img)
            ui.update_3d_image(img)
            if(ui.checkclose()):
                exit()
        if not FLAGS.hide_debug:
            print("\n")
            #print('Got %s poses in %.3f seconds'%(len(detections), time.time() - start_time))
            print('-------------')

        if FLAGS.video:
            cv2.waitKey(1)
        else:
                key = cv2.waitKey(0) 
                if key == 27:
                    exit()
                elif key != 32:
                    continue

if __name__ == '__main__':
    main()
