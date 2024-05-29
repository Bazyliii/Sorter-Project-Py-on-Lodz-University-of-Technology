import cv2
import numpy as np

# capture = cv2.VideoCapture(r"C:/Users/jaros/Downloads/record.avi")

# cv2.namedWindow('image')
# def nothing(x):
#     pass

def main():
    capture = cv2.VideoCapture(0)
    fps = capture.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)
    # delay = 1
    # cv2.createTrackbar('low_1','image',0,255,nothing)
    # cv2.createTrackbar('low_2','image',38,255,nothing)
    # cv2.createTrackbar('low_3','image',123,255,nothing)
    # cv2.createTrackbar('high_1','image',28,255,nothing)
    # cv2.createTrackbar('high_2','image',255,255,nothing)
    # cv2.createTrackbar('high_3','image',255,255,nothing)
    temp = []
    while True:
        _, frame = capture.read()
        # frame = cv2.imread(r"C:/Users/jaros/Downloads/dataset/obraz (18).png")
        frame = frame[200:490, 200:300]
        blur = cv2.GaussianBlur(frame, (5, 5), 0)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        # low_1 = cv2.getTrackbarPos('low_1','image')
        # low_2 = cv2.getTrackbarPos('low_2','image')
        # low_3 = cv2.getTrackbarPos('low_3','image')
        # high_1 = cv2.getTrackbarPos('high_1','image')
        # high_2 = cv2.getTrackbarPos('high_2','image')
        # high_3 = cv2.getTrackbarPos('high_3','image')
        
        # lower_orange = np.array([low_1, low_2, low_3])
        # upper_orange = np.array([high_1, high_2, high_3])
        # mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        lower_orange = np.array([0, 38, 123])
        upper_orange = np.array([58, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel)
        
        # ret, thresh = cv2.threshold(gray, 127, 255, 0)
        
        contours, _ = cv2.findContours(mask, 1, 2)
        if len(contours) < 1:
            if len(temp) > 0:
                print(temp[int(len(temp)/2)])
            temp = []
        for cnt in contours:
            epsilon = 0.015*cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,epsilon,True)
            match len(approx):
                case 12:
                    temp.append("Cross")
                case 6:
                    temp.append("Tetrahedron")                    
                case 4:
                    temp.append("Cube")
                case 3:
                    temp.append("Triangle")
                    
            
            # x = approx.ravel()[0]
            # y = approx.ravel()[1]
            
            # cv2.putText(frame, str(len(approx)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)



        ####### ORANGE DETECTION ########
        # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # lower_orange = np.array([5, 50, 50])
        # upper_orange = np.array([25, 255, 255])
        # mask = cv2.inRange(hsv, lower_orange, upper_orange)
        # kernel = np.ones((5, 5), np.uint8)
        # mask = cv2.erode(mask, kernel)
     
        # _, contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(frame, contours, -1, (0, 255, 0), 6)    

        # print("Number of Contours found = " + str(len(contours)))
        cv2.imshow("mask", mask)
        ####### ORANGE DETECTION ######## 
        cv2.imshow("frame", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break


# def test():
#     capture = cv2.VideoCapture(0, cv2.CAP_MSMF)
#     # capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
#     capture.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
#     capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
#     height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
#     print(height, width)
#     while True:
#         _, frame = capture.read()
#         cv2.imshow("frame", frame)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

if __name__ == "__main__":
    main()
    # test()
