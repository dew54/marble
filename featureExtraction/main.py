    

import os
import cv2
import csv
import numpy as np
def resize(image, w = 100, h = 100):
    final_image = cv2.resize(image, (w, h))
    return final_image




def extract_DESC(image, circle_info):        

    image_descriptors = []
    descriptors = []
    keypoints = []
    for info in circle_info:
        image_with_sift = image
        
        print(info)
        center_x, center_y = info  #info['Center']#map(int, info['Center'].strip('()').split(','))  # Convert the comma-separated string to tuple
        radius = 90#int(info['Radius'])

        scale_factor = 1 + 0.004 * radius  # Adjust the scaling factor as needed

        # Calculate the enlarged region of interest
        x1, y1 = center_x - int(scale_factor * radius), center_y - int(scale_factor * radius)
        x2, y2 = center_x + int(scale_factor * radius), center_y + int(scale_factor * radius)

        # radius = scale_factor * radius
        img_desc = []
        # Check if the coordinates are within the image boundaries
        if center_x - radius >= 0 and center_y - radius >= 0 and center_x + radius <= image.shape[1] and center_y + radius <= image.shape[0]:
            # Extract the region of interest (ROI) based on the circle information


            roi = image_with_sift[y1:y2, x1:x2]

            if roi is not None and roi.shape[0] > 0 and roi.shape[1] > 0:
                
                # Extract SIFT keypoints and descriptors for the ROI
                # gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                # keypoints, descriptors = sift.detectAndCompute(gray_roi, None)
                
                orb = cv2.ORB_create()

                keypoints, descriptors = orb.detectAndCompute(roi, None)

                if descriptors is not None:
                    keypoints_and_descriptors = list(zip(keypoints, descriptors))

                    keypoints_and_descriptors.sort(key=lambda kp_desc: kp_desc[0].response, reverse=True)

                    # Select the top N keypoints and descriptors
                    num_selected_keypoints = 10
                    selected_keypoints, selected_descriptors = zip(*keypoints_and_descriptors[:num_selected_keypoints])

                    # Convert selected_descriptors to a numpy array
                    descriptors = list(selected_descriptors)

                    keypoints = selected_keypoints
                    print(len(keypoints))



                # Filter keypoints and descriptors based on the response threshold
                # keypointsResp = [kp.response for kp in keypoints ]
                # print(keypointsResp)
                # descriptors = descriptors[keypoints.index(kp) for kp in filtered_keypoints]

                # roi_with_keypoints = add_sift_to_image(keypoints, roi)
                # cv2.imshow("dd", roi_with_keypoints)
                # cv2.waitKey(0)

                # print(descriptors)



    return keypoints, descriptors

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

def add_sift_to_image( keypoints, image):
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def hough_circle_transform(image, min_radius, max_radius, threshold=100):

        center_coordinates = []

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise (use gray_image instead of image)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
        

    # Apply Hough Circle Transform
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

        # Process the results
        if circles is not None:
            for circle in circles[0, :]:
                x, y, radius = circle
                center_coordinates.append((int(x), int(y)))

                # circles = np.uint16(np.around(circles))

            return center_coordinates

        return []

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

    # # Draw circles at the center of each stain
    # for center in center_coordinates:
    #     cv2.circle(image, center, 5, (0, 0, 255), -1)

    # cv2.imshow('Center of Stains', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return center_coordinates
def extractDetectedCircles(image, min_radius, max_radius ):
    # Call the Hough Circle Transform function
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    detected_circles = hough_circle_transform(image, min_radius, max_radius, 100)
    return detected_circles  


def extractCirclesInfo(image, min_radius, max_radius):


    detected_circles = extractDetectedCircles(prod_image, min_radius, max_radius)


    circle_info = []
    image = cv2.cvtColor(resize(image), cv2.COLOR_RGB2GRAY)
    prod_image = cv2.cvtColor(prod_image, cv2.COLOR_GRAY2RGB)

        # Draw the detected circles on the image and save center and radius
    for i, circle in enumerate(detected_circles, start=1):
        center = (circle[0], circle[1])
        radius = circle[2]
        
        cv2.circle(prod_image, center, radius, (0, 255, 0), 2)

        # Save center and radius information
        circle_info.append({"Circle": i, "Center": center, "Radius": radius})

        # FeatureExtraction.process_image(luminance_image, csv_file, output_image_path, counter)
    cv2.imwrite("Circles.jpeg", prod_image)
    return circle_info
    
def sliding_correlation(reference, target):
    # Convert images to grayscale
    # ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    
    # Calculate correlation map
    correlation_map = cv2.matchTemplate(target_gray, reference, cv2.TM_CCOEFF_NORMED)
    
    # Normalize correlation map to [0, 255] for visualization
    correlation_map = cv2.normalize(correlation_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Convert correlation map to color image (all channels have the same values)
    correlation_colormap = cv2.cvtColor(correlation_map, cv2.COLOR_GRAY2BGR)
    
    # Resize the correlation colormap to match the dimensions of the target image
    correlation_colormap = cv2.resize(correlation_colormap, (target.shape[1], target.shape[0]))
    
    # Combine the target image and the resized correlation colormap
    result_image = cv2.addWeighted(target, 1, correlation_colormap, 0.5, 0)
    
    return correlation_colormap




if __name__ == "__main__":

    typ = "train"
    spec = "larvae"


    # path = "./data_augmented/larvae"# + "/_image_003.png"
    path = "./" + typ + "/" + spec

    refPath = "ref.png"

    ref = cv2.imread(refPath, cv2.IMREAD_GRAYSCALE)


    images = loadImages(path)

    all_desc = []

    for image in images:

        # image = resize(image, 500, 500)

        # image = cv2.fastNlMeansDenoisingColored(image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create a CLAHE object and apply CLAHE to the grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(gray_image)



        # image = cv2.medianBlur(image, 3)

        # image = cv2.GaussianBlur(image, (3, 3), sigmaX=0)




        correlationMap = sliding_correlation(resize(ref,200,210), image)


        circles_info =  findCenters(correlationMap)


        keypoints, image_desc = extract_DESC(clahe_image, circles_info)
        if image_desc is not None:
            for desc in image_desc:
                descList = desc.tolist()
                descList.insert(0, spec)
                all_desc.append(descList)





        

        # gray_roi = unsharpMask(gray_roi)

        # gray_roi = cv2.equalizeHist(gray_roi)

        # cv2.imshow("dd",gray_roi)
        # kernel_size = (11, 11)  # Specify the size of the Gaussian kernel
        # image = cv2.GaussianBlur(image, kernel_size, 0)






        # sift = cv2.SIFT_create()
        # keypoints, descriptors = sift.detectAndCompute(gray_roi, None)

##################################################################################
        # orb = cv2.ORB_create()
        # keypoints, descriptors = orb.detectAndCompute(clahe_image, None)

        # if descriptors is not None:

        #     keypoints_and_descriptors = list(zip(keypoints, descriptors))

        #     keypoints_and_descriptors.sort(key=lambda kp_desc: kp_desc[0].response, reverse=True)

        #     # Select the top N keypoints and descriptors
        #     num_selected_keypoints = 10
        #     selected_keypoints, selected_descriptors = zip(*keypoints_and_descriptors[:num_selected_keypoints])

        #     # Convert selected_descriptors to a numpy array
        #     descriptors = list(selected_descriptors)
##################################################################################
        
        # # Create an AKAZE object
        # akaze = cv2.AKAZE_create()

        # # Detect keypoints and compute descriptors
        # keypoints, descriptors = akaze.detectAndCompute(image, None)
        

        # brisk = cv2.BRISK_create()
        # keypoints, descriptors = brisk.detectAndCompute(image, None)


        # descriptors_normalized = cv2.normalize(descriptors, None, norm_type=cv2.NORM_L2)
##################################################################################
        
            # for desc in descriptors:
            #     # print(len(desc))
            #     desc = desc.tolist()
            #     desc.insert(0, spec)
            #     all_desc.append(desc)
##################################################################################



    fileName = typ + "_" + spec + "_ORB.csv"

    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file)
        for desc in all_desc:
            writer.writerow(desc)
    