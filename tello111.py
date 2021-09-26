'''
import cv2
import time
from djitellopy import Tello
import numpy as np

width = 1280  # WIDTH OF THE IMAGE
height = 720  # HEIGHT OF THE IMAGE
deadZone = 100

startCounter = 0

# CONNECT TO TELLO
me = Tello()
me.connect()


me.streamoff()
me.streamon()


img_counter = 0
frame_set = []
start_time = time.time()
while True:


    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))

    capture = cv2.VideoCapture(0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start_time >= 3: #<---- Check if 5 sec passed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, myFrame)
        print("{} written!".format(img_counter))
        start_time = time.time()
'''

# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from torchvision import transforms
from PIL import Image
from PIL import ImageDraw
import copy
from matplotlib.pyplot import imshow

"from djitellopy import Tello"

FILE = Path(__file__).resolve()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, check_requirements, check_suffix, colorstr, is_ascii, \
    non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, \
    save_one_box
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, load_classifier, time_sync
from djitellopy import Tello
from roi1 import follow

# CONNECT TO TELLO
# me = Tello()
# me.connect()
#
# me.streamoff()
# me.streamon()

# tello_vedio = cv2.VideoCapture('udp://0.0.0.0:11111')
# ret, frame = tello_vedio.read()


# frame_read = me.get_frame_read()
# myFrame = frame_read.frame

@torch.no_grad()

def yellow_hsv(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([12, 75, 160])
    upper_yellow = np.array([56, 255, 255])
    img_th = cv2.inRange(image_hsv, lower_yellow, upper_yellow)
    return img_th

def run(weights='last.pt',  # model.pt path(s)   'yolov5s.pt'
        # source=None,  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels+
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        source=None
        ):

    webcam = False

    me = Tello()
    me.connect()

    me.streamoff()
    me.streamon()



    me.for_back_velocity = 0
    me.left_right_velocity = 0
    me.up_down_velocity = 0
    me.yaw_velocity = 0  # 기울기

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    w = weights[0] if isinstance(weights, list) else weights
    classify, suffix, suffixes = False, Path(w).suffix.lower(), ['.pt', '.onnx', '.tflite', '.pb', '']
    check_suffix(w, suffixes)  # check weights have acceptable suffix
    pt, onnx, tflite, pb, saved_model = (suffix == x for x in suffixes)  # backend booleans
    stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
    if pt:
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if half:
            model.half()  # to FP16
        if classify:  # second-stage classifier
            modelc = load_classifier(name='resnet50', n=2)  # initialize
            modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()
    elif onnx:
        check_requirements(('onnx', 'onnxruntime'))
        import onnxruntime
        session = onnxruntime.InferenceSession(w, None)
    else:  # TensorFlow models
        check_requirements(('tensorflow>=2.4.1',))
        import tensorflow as tf
        if pb:  # https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
            def wrap_frozen_graph(gd, inputs, outputs):
                x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped import
                return x.prune(tf.nest.map_structure(x.graph.as_graph_element, inputs),
                               tf.nest.map_structure(x.graph.as_graph_element, outputs))

            graph_def = tf.Graph().as_graph_def()
            graph_def.ParseFromString(open(w, 'rb').read())
            frozen_func = wrap_frozen_graph(gd=graph_def, inputs="x:0", outputs="Identity:0")
        elif saved_model:
            model = tf.keras.models.load_model(w)
        elif tflite:
            interpreter = tf.lite.Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            int8 = input_details[0]['dtype'] == np.uint8  # is TFLite quantized uint8 model
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    ascii = is_ascii(names)  # names are ascii (use PIL for UTF-8)

    #############################################################################################
    #여기서 부터 프레임 받아서 처리

    # 이미지 전처리 -> 한번만 정의 후 프레임에 대해서 전처리
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])

    me.takeoff()
    me.move_up(50)

    while True:
        frame_read = me.get_frame_read()
        myFrame = frame_read.frame
        img_cv = copy.deepcopy(myFrame)

        myFrame = Image.fromarray(myFrame)
        image = copy.deepcopy(myFrame)
        image = image.resize((640, 640))
        # preprocessing input Image
        myFrame = transform(myFrame)
        # 필요함 그냥 넣으셈
        myFrame = torch.unsqueeze(myFrame, 0)



        dataset = myFrame  # 수정
        bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs
        dataset = dataset.to(device)
        pred = model(dataset)[0]


        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        bbox = pred
        image_draw = ImageDraw.Draw(image, mode="RGB")
        bbox[0] = bbox[0].cpu()

        ###################################################################
        bi_yellow = yellow_hsv(img_cv)
        yvalue_th = np.where(bi_yellow[:, :] == 255)
        if (np.sum(yvalue_th[0]) == 0 or np.sum(yvalue_th[1]) == 0):
            print("no color")
        else:
            ymin_x1 = np.min(yvalue_th[1])
            ymax_x1 = np.max(yvalue_th[1])

            ycenter_x1 = int((ymin_x1 + ymax_x1) / 2)

            print(f'ymin_x1 = {ymin_x1} , ymax_x1 = {ymax_x1}')



        #########################################################3

        if bbox[0].size() == torch.Size([0, 6]):
            result = image #원본 사진
            print('can not find')
        else:
            # print(bbox[0][0])
            bbox1 = bbox[0][0].cpu().numpy()
            #print(int(bbox[0])) #-> 532 픽셀좌표로 뜬다 확인함.
            image_draw.rectangle(((bbox1[0], bbox1[1]), (bbox1[2], bbox1[3])), outline=(0, 0, 255), width=4)
            result = image #상자 그려진 사진
            dir_1 = follow( bbox1[0] , bbox1[1],bbox1[2],bbox1[3])
            # print('dir_check')

            if dir_1 == 1:
                me.move_left(20)
                print('go left')
            elif dir_1 == 2:
                me.move_right(20)
                print('go right ')
            elif dir_1 == 3:
                me.move_back(20)
                print('go back')
            elif dir_1 == 4:
                me.move_forward(20)
                print('go forward')
            else:
                print('stay')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                me.land()
                print('land')

        ##########################################################

        if (np.sum(yvalue_th[0]) == 0 or np.sum(yvalue_th[1]) == 0):
            print('no color')
        else:
            if (ycenter_x1 + 50 < int((bbox1[0] + bbox1[2]) / 2)):
                print("오른쪽으로 기울어짐 왼쪽으로 이동하세요.")
            elif (ycenter_x1 - 50 > int((bbox1[0] + bbox1[2]) / 2)):
                print("왼쪽으로 기울어짐 오른쪽으로 이동하세요.")
            else:
                print("안정적으로 보행 중입니다.")

        ########################################################

        result = np.asarray(result)

        # 중점 라인
        cv2.line(result, (ycenter_x1 - 30, ymin_y1), (ycenter_x1 - 30, ymax_y1), (255, 0, 0), 5, cv2.LINE_AA)
        cv2.line(result, (ycenter_x1 + 30, ymin_y1), (ycenter_x1 + 30, ymax_y1), (255, 0, 0), 5, cv2.LINE_AA)

        #격자 생성
        cv2.line(result, (int(200), 0), (int(200), 640), (255, 255, 0), 3)
        cv2.line(result, (int(440), 0), (int(440), 640), (255, 255, 0), 3)
        cv2.circle(result, (int(640 / 2), int(640 / 2)), 5, (0, 0, 255), 5)
        cv2.line(result, (0, int(200)), (640, int(200)), (255, 255, 0), 3)
        cv2.line(result, (0, int(440)), (640, int(440)), (255, 255, 0), 3)


        # cv2.line(result, (320,320), ((bbox1[0] + bbox1[2]) / 2),((bbox1[1] + bbox1[3]) / 2), (0, 0, 255), 3)
        # 실시간으로 받아오게끔
        cv2.imshow('1', result)







    # return pred


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='last.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=None, help='file/dir/URL/glob, 0 for webcam') # source -> None
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.15, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))

    run(**vars(opt))


if __name__ == "__main__":

    opt = parse_opt()
    main(opt)