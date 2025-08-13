import numpy as np
import cv2
import imutils

# Parameters
blur = 21
canny_low = 15
canny_high = 150


def getHoughLines(img, edges: np.ndarray):
    """ Finds all straight lines in an image 
    Returns: the image of line detections
    """
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    # minimum number of votes (intersections in Hough grid cell)
    threshold = 15
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Lines are organized x1,y1,x2,y2 (start point, end point)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

    # Draw the lines on the  image
    lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
    return line_image

def getHoughCircles(img):
    """ Finds all circles
    Returns: the image of circle detections
    """
    kernel = np.ones((5,5),np.uint8)

    erosion = cv2.erode(img,kernel,iterations = 8)
    grey = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    minDist = 100
    param1 = 30  # 500
    param2 = 40  # 200 #smaller value-> more false circles
    minRadius = 10
    maxRadius = 500  # 10

    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
    return img, circles


def getBlobsSimple(img):
    params = cv2.SimpleBlobDetector_Params()

    # # Set Area filtering parameters
    # params.filterByArea = True
    # params.minArea = cv2.getTrackbarPos("Blob_Area","Parameters")
    
    # # Set Circularity filtering parameters
    # params.filterByCircularity = True
    # params.minCircularity = cv2.getTrackbarPos("Blob_Circularity","Parameters")/100
    
    # # Set Convexity filtering parameters
    # params.filterByConvexity = True
    # params.minConvexity = cv2.getTrackbarPos("Blob_Convexity","Parameters")/100
        
    # # Set inertia filtering parameters
    # params.filterByInertia = True
    # params.minInertiaRatio = cv2.getTrackbarPos("Blob_Inertia","Parameters")/100
    
    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
        
    # Detect blobs
    return detector.detect(img)

def get_blobs(img):
    """ Returns the contour that contains the circle, using Hough Circles and Canny edge detection to get the contours """
    new_img, circles = getHoughCircles(img)
    # Get contours
    contours = getContoursWithoutResize(img, 8, 10)
    contours = imutils.grab_contours(contours)
    if circles is not None:
        for circle in circles[0, :]:
            # check if the circle is in any of the contours
            print("circle: ", circle)
            center = (circle[0], circle[1])
            for c in contours:
                # use point polygon test to check if an (x,y) point is within a contour
                if cv2.pointPolygonTest(c, center, False) == 1.0:
                    rectangle_min = cv2.minAreaRect(c)
                    box_min = np.int0(cv2.boxPoints(rectangle_min))
                    cv2.polylines(img,[box_min],True,(255,0,0),2) # draw rectangle in blue color
                    return img
    return None      
        
def getContoursAndResizeImg(img, erosion_iterations, dilate_iterations):
    """ Finds contours in a binary image
    Args:
    erosion_iterations: number from 1-10 for the iterations to perform image erosion. Generally a higher number = less noise
    dilate_iterations: number from 1-10 for the iterations to perform image dilate. Generally a higher number = greater size of edges
    Returns:
    The image contours """
    edges = getEdges(img, erosion_iterations, dilate_iterations)

    # Get the contours of the image
    contours = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def getContoursWithoutResize(img, erosion_iterations, dilate_iterations):
    """ Finds contours in a binary image
    Args:
    erosion_iterations: number from 1-10 for the iterations to perform image erosion. Generally a higher number = less noise
    dilate_iterations: number from 1-10 for the iterations to perform image dilate. Generally a higher number = greater size of edges
    Returns:
    The image contours """
    edges = getEdgesWithoutResize(img, erosion_iterations, dilate_iterations)

    # Get the contours of the image
    contours = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def setRectangleProperties(x, y, w, h):
    """ Set rectangle properties. CV2 defines the coordinate system to increase from the top left of the image.

    Args:
    x: x position of the top left corner
    y: x position of the top left corner
    w: width of the rectangle
    h: height of the rectangle
    rx: x position of the bottom right corner
    ry: y position of the bottom right corner
    """
    data = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "rx": x + w,
        "ry": y + h
    }
    return data

def getEdges(img, erosion_iterations, dilate_iterations):
    """ Perform Canny edge detection processing the image first to have less noise 
    Args:
    erosion_iterations: number from 1-10 for the iterations to perform image erosion. Generally a higher number = less noise
    dilate_iterations: number from 1-10 for the iterations to perform image dilate. Generally a higher number = greater size of edges
    Returns:
    The image edges from Canny edge detection """
    # Setting parameter values
    kernel = np.ones((5,5),np.uint8)
    t_lower = 60  # Lower Threshold
    t_upper = 255  # Upper threshold

    erosion = cv2.erode(img,kernel,iterations = erosion_iterations)
    dilate = cv2.dilate(erosion,kernel,iterations = dilate_iterations)
    cv2.bitwise_not(dilate, dilate)
    image = dilate
    resized = imutils.resize(image, width=600)
    ratio = image.shape[0] / float(resized.shape[0])
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    thresh = cv2.threshold(blurred, t_lower, t_upper, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(blurred, t_lower, t_upper)
    return edges

def getEdgesWithoutResize(img, erosion_iterations, dilate_iterations):
    """ Perform Canny edge detection processing the image first to have less noise 
    Args:
    erosion_iterations: number from 1-10 for the iterations to perform image erosion. Generally a higher number = less noise
    dilate_iterations: number from 1-10 for the iterations to perform image dilate. Generally a higher number = greater size of edges
    Returns:
    The image edges from Canny edge detection """
    # Setting parameter values
    kernel = np.ones((5,5),np.uint8)
    t_lower = 60  # Lower Threshold
    t_upper = 255  # Upper threshold

    erosion = cv2.erode(img,kernel,iterations = erosion_iterations)
    dilate = cv2.dilate(erosion,kernel,iterations = dilate_iterations)
    cv2.bitwise_not(dilate, dilate)
    image = dilate
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    thresh = cv2.threshold(blurred, t_lower, t_upper, cv2.THRESH_BINARY)[1]
    edges = cv2.Canny(blurred, t_lower, t_upper)
    return edges

def get_body(image):
    """ Returns: the top 5 shapes makng up the body rectangle value in format 
    {"x": x, "y": y, "w": w, "h": h, "rx": rx, "ry": ry}"""
    contours = getContoursWithoutResize(image, 5, 10)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5] # get the sorted 5 largest contours
    results = []
    for c in contours:
        (x,y,w,h) = cv2.boundingRect(c)
        body_rc = setRectangleProperties(x,y,w,h)
        results.append(body_rc)
    return results
    
def detect_shapes(image):
    """Workflow, getting contours, sorting top 3, getting circles for wheels"""
    # get_body(image)
   
    # return contours

def main():
    """Use test images of busses and cars to perform image processing on (using Canny),
    then overlay the findings over the original image to visualize the results"""
    images_to_process = ["2024-hyundai.jpg", "car-sports.jpeg"]
    originals = []
    edges = []
    contours = []
    for vi in images_to_process:
        # open image
        vehicle_img = cv2.imread(vi, cv2.IMREAD_GRAYSCALE)
        # originals
        originals.append(vehicle_img)
        # get edges
        edges.append(getEdges(vehicle_img, 4, 10))
        # get contours

        results = get_body(vehicle_img)
        curr_image = vehicle_img
        for x in results:
            cv2.rectangle(curr_image, (x["x"],x["y"]), (x["x"]+x["w"],x["y"]+x["h"]), (0,255,0), 2)
        contours.append(curr_image)

    # visualise
    for x in range(len(images_to_process)):
        cv2.imshow('Original ' + images_to_process[x], originals[x])
        cv2.imshow('Edge Detections ' + images_to_process[x], edges[x])
        # cv2.drawContours(originals[x], contours[x], -1, (0,255,0), 3)
        cv2.imshow('Intermediate Contours ' + images_to_process[x], contours[x])


if __name__ == "__main__":
    main()
