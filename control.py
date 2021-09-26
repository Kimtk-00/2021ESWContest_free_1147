graph = {
    'A': {'B': 110},
    'B': {'A': 110, 'C': 50, 'D': 30},
    'C': {'B': 50},
    'D': {'B': 30, 'E': 50, 'F': 110},
    'E': {'D': 50},
    'F': {'D': 110, 'G': 40, 'H': 100},
    'G': {'F': 40},
    'H': {'F': 100, 'I': 40, 'J': 240},
    'I': {'H': 40},
    'J': {'H': 240}
}

import heapq
from djitellopy import Tello
import time
import cv2
import numpy as np

tello = Tello()

def crosswalk(image):
    xw = yw = cnt = 0
    lower_white = np.array([0, 0, 245])
    upper_white = np.array([180, 255, 255])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_white = cv2.inRange(image, lower_white, upper_white)
    img_result2 = cv2.bitwise_and(image, image, mask=mask_white)
    img_white_gray = img_result2[:, :, 2]
    img_blur = cv2.GaussianBlur(img_white_gray, ksize=(5, 5), sigmaX=0)
    img_thresh = cv2.adaptiveThreshold(
        img_blur, maxValue=255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV, blockSize=19, C=9)
    contours, _ = cv2.findContours(
        img_thresh,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_SIMPLE
    )


    for contour in contours:
        img_thresh = cv2.drawContours(img_thresh, contour, -1, (255,255,255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        image = cv2.drawContours(image, contour, -1, (255,255,255), 2)
        x, y, w, h = cv2.boundingRect(contour)
        print(x,y,w,h)
        if w in range(80, 150) and h in range(20, 70):
            cv2.rectangle(img_thresh, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)
            xw += (x+(w//2))
            cnt+=1
    cv2.imshow("test", img_thresh)
    cv2.imshow("original", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    if cnt == 0:
        cv2.imwrite("test_cross.jpg", img_thresh)
        return "no cross"
    else:
        cv2.imwrite("test_cross.jpg", img_thresh)
        return xw//cnt

def check_rotate(path):
    if path == "B > C" or path == "D > E" or path == "H > I":
        tello.rotate_clockwise(90)
        print("see right")

    elif path == "F > H" or path == "H > J":
        tello.rotate_counter_clockwise(90)
        print("see left")

def check_crosswalk(path):

    if path == "B > D" or path == "F > H":

        image = frame_read.frame
        image = cv2.resize(image, (640, 640))
        cv2.imwrite("C:/Users/eee85/Desktop/testing.jpg", image)
        x = crosswalk(image)
        if x == 'no cross':
            print("no cross")
        elif x > 320:
            leng = (x - 320) // 5

            if leng < 5:
                print("cross crosswalk")
            else:
                print("go right %dcm" %leng)

        elif x < 320:
            leng = (320 - x) // 5

            if leng < 5:
                print("cross crosswalk")
            else:
                print("go left %dcm" %leng)




def dijkstra(graph, start, end):
    d = {vertex: [float('inf'), start] for vertex in graph}
    d[start] = [0, start]
    queue = []
    heapq.heappush(queue, [d[start][0], start])

    while queue:
        current_d, current_vertex = heapq.heappop(queue)
        if d[current_vertex][0] < current_d:
            continue
        for adjacent, weight in graph[current_vertex].items():
            dis = current_d + weight
            if dis < d[adjacent][0]:
                d[adjacent] = [dis, current_vertex]
                heapq.heappush(queue, [dis, adjacent])

    path = end
    path_output = list(end)
    while d[path][1] != start:
        path_output.append(d[path][1])
        path = d[path][1]
    path_output.append(start)
    return path_output

path = list(reversed(dijkstra(graph, "A", "F")))
print(path)
print()

tello.connect()
tello.streamon()




tello.takeoff()
tello.move_up(50)
frame_read = tello.get_frame_read()

for i in range(len(path)-1):
    
    print(path[i] + ' > ' + path[i+1] + "  ||  " + str(graph[path[i]][path[i+1]]))
    check_rotate(str(path[i]) + ' > ' + str(path[i+1]))
    check_crosswalk(str(path[i]) + ' > ' + str(path[i+1]))



    tello.move_forward(int(graph[path[i]][path[i+1]]))

    time.sleep(1)

tello.land()
tello.streamoff()