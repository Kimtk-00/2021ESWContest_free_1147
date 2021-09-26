global imgContour
global dir;


def follow( x , y ,h ,w):
    global dir

    x , y , w, h = x , y , h , w
    cx = int((x + w) / 2)  # CENTER X OF THE OBJECT
    cy = int((y + h) / 2)  # CENTER X OF THE OBJECT

    if (cx <int(200)):
        dir = 1
    elif (cx > int(440)):
        dir = 2
    elif (cy < int(200)):
        dir = 3
    elif (cy > int(440)):
        dir = 4

    else:
        dir=0

    return dir
