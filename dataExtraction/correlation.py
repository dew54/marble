import cv2
import os
import numpy as np

def resize(image, w = 100, h = 100):
    final_image = cv2.resize(image, (w, h))
    return final_image

def findBlobs(img):
    # Set up the SimpleBlobDetector with default parameters
    params = cv2.SimpleBlobDetector_Params()

    # Set the threshold
    params.minThreshold = 0
    params.maxThreshold = 20000

    # Set the area filter
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 10000

    # Set the circularity filter
    params.filterByCircularity = True
    params.minCircularity = 0.01
    params.maxCircularity = 1

    # Set the convexity filter
    params.filterByConvexity = True
    params.minConvexity = 0.010
    params.maxConvexity = 1

    # Set the inertia filter
    params.filterByInertia = True
    params.minInertiaRatio = 0.01
    params.maxInertiaRatio = 1

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(img)

    # Draw detected blobs as red circles
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show the image with detected blobs
    cv2.imshow("Blobs", img_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def findCentres(image):
    edges = cv2.Canny(image, threshold1=70, threshold2=100)

    edges = cv2.medianBlur(edges, 3)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




    image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    cnt = contours[areas.index(sorted_areas[-1])]
    
    (x, y), radius = cv2.minEnclosingCircle(cnt)

    center = (int(x), int(y))
    radius = int(radius)

    cv2.circle(image, center, radius, (1), -1)

    cv2.imshow("FINAL",image)
    cv2.waitKey(0)

def sliding_correlation(reference, target):
    # Convert images to grayscale
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # Calculate correlation map
    correlation_map = cv2.matchTemplate(target_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
    
    # Normalize correlation map to [0, 255] for visualization
    correlation_map = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Convert correlation map to color image (all channels have the same values)
    correlation_colormap = cv2.cvtColor(correlation_map, cv2.COLOR_GRAY2BGR)
    
    # Resize the correlation colormap to match the dimensions of the target image
    correlation_colormap = cv2.resize(correlation_colormap, (target.shape[1], target.shape[0]))
    
    # Combine the target image and the resized correlation colormap
    result_image = cv2.addWeighted(target, 1, correlation_colormap, 0.5, 0)
    
    return correlation_colormap


def loadImages(input_path):

    images = []
    # Check if the input path is a file or a directory
    if os.path.isfile(input_path):  # Single image file
        image = cv2.imread(input_path)
        if image is not None:
            images.append(image)
    elif os.path.isdir(input_path):  # Directory containing multiple images
        for filename in os.listdir(input_path):
            image_path = os.path.join(input_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images.append(image)

    return images

def findCenters(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Threshold the image to separate stains from the background
    _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the center of each contour
    center_coordinates = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center_coordinates.append((cX, cY))

    # Draw circles at the center of each stain
    # for center in center_coordinates:
    #     cv2.circle(image, center, 5, (0, 0, 255), -1)

    # Display the result
    # cv2.imshow('Center of Stains', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return center_coordinates





if __name__ == "__main__":
      # path = "./dataset/oyster_larvae/single/image_003.png"
    path = "./dataset/oyster_larvae/single" + "/image_003.png"
    # path = "./dataset/other_planktons/single"

    images = loadImages(path)
    reference_image = cv2.imread('./ref.png')
    reference_image = resize(reference_image, 200, 200)



    counter = 000

    for image in images:
        target_image = image
        target_image = resize(target_image, 1280, 720)
        
        window_size = (reference_image.shape[1], reference_image.shape[0])  # Use the size of the reference image
        
        result_image = sliding_correlation(reference_image, target_image)
        # result_image[result_image < 200] = 0
        cv2.imshow('Original', target_image)

        cv2.imshow('Correlation Landscape', result_image)
        
        cv2.waitKey(0)

        findCenters(result_image)
        
        cv2.destroyAllWindows()
