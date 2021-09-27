import cv2
import os
import time

vcnt = 0
cnt_capture = 0
cut_frame_cnt = 0
start_time = time.time()
cut_frame = 5 #장 수

video_path = "C:/Users/LHS/Videos/"
saveimg_path = "C:/Users/LHS/save/"
filePath = os.path.join(video_path, '%d.mp4' % vcnt)

while os.path.isfile(filePath):
    filePath = os.path.join(video_path, '%d.mp4' % vcnt)
    cap = cv2.VideoCapture(filePath)

    while True:
        ret, frame = cap.read()

        if not ret:
            vcnt += 1
            break

        if cut_frame_cnt == cut_frame:
            cv2.imwrite(saveimg_path + "%d.jpg" % cnt_capture, frame)
            print(str(vcnt) +" video " + str(cnt_capture) + " cap saved // elapsed time is " + str(time.time()-start_time))
            cut_frame_cnt = 0
            cnt_capture += 1

        cut_frame_cnt += 1

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

    cap.release()

cv2.destroyAllWindows()