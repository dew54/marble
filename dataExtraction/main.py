import cv2
import os
import numpy as np
from correlation import *




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

def compute_barycenter(image):
    # Load the image using OpenCV
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Get the dimensions of the image
    # image = preProcess(image)
    height, width = image.shape
    
    # Initialize variables for weighted sums and total weight
    weighted_sum_x = 0
    weighted_sum_y = 0
    total_weight = 0
    
    # Iterate over each pixel
    for x in range(width):
        for y in range(height):
            pixel_intensity = image[y, x]
            
            # Accumulate weighted sums and total weight
            weighted_sum_x += x * pixel_intensity
            weighted_sum_y += y * pixel_intensity
            total_weight += pixel_intensity
    
    # Calculate barycenter coordinates
    barycenter_x = weighted_sum_x / total_weight
    barycenter_y = weighted_sum_y / total_weight
    
    return barycenter_x, barycenter_y


def hough_circle_transform(image, min_radius, max_radius, threshold=10):

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise (use gray_image instead of image)
    blurred_image = cv2.GaussianBlur(image, (9, 9), 2)
    

# Apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius)

    # Process the results
    if circles is not None:
        circles = np.uint16(np.around(circles))

        return circles[0, :]

    return []

def preProcess(image):

    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



    # Calculate the gradient using Sobel operator (X and Y directions)
    gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)

    # Combine gradients to get the magnitude
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    
    gradient_magnitude_uint8 = cv2.normalize(gradient_magnitude, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    edges = cv2.Canny(gradient_magnitude_uint8, threshold1=70, threshold2=100)

    edges = cv2.medianBlur(edges, 3)

    return edges


    # contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)




    # image_with_contours = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # areas = [cv2.contourArea(c) for c in contours]
    # sorted_areas = np.sort(areas)
    # cnt = contours[areas.index(sorted_areas[-1])]
    
    # (x, y), radius = cv2.minEnclosingCircle(cnt)

    # center = (int(x), int(y))
    # radius = int(radius)

    # cv2.circle(image, center, radius, (1), -1)




    # cv2.imshow("IMG",edges)
    # cv2.waitKey(0)
    # cv2.imshow("IMG",image)
    # cv2.waitKey(0)



def findCircles(image, minR, maxR):
    detected_circles = hough_circle_transform(image, minR, maxR)

    for i, circle in enumerate(detected_circles, start=1):
        center = (circle[0], circle[1])
        radius = circle[2]
        
        cv2.circle(image, center, radius, (0, 255, 0), 2)

        # Save center and radius information
    cv2.imshow("img",image)
    cv2.waitKey(0)

def crop_around_barycenter(image, target_size):
    
    barycenter_x, barycenter_y = compute_barycenter(image)
    
    crop_x = int(max(0, barycenter_x - target_size[0] / 2))
    crop_y = int(max(0, barycenter_y - target_size[1] / 2))
    
    cropped_image = image[crop_y:crop_y + target_size[1], crop_x:crop_x + target_size[0]]
    
    return cropped_image

def resize(image, w = 100, h = 100):
    final_image = cv2.resize(image, (w, h))
    return final_image




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


def findCentroids(image):
    _, thresholded = cv2.threshold(image, 170, 200, cv2.THRESH_BINARY)  # Apply thresholding to create binary image


    cv2.imshow('Centroids', thresholded)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
    # Calculate centroid of each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            centroid = (centroid_x, centroid_y)
            
            # Draw a circle at the centroid
            cv2.circle(image, centroid, 5, (0, 0, 255), -1)
    cv2.imshow('Centroids', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows




if __name__ == "__main__":

    path = "./dataset/oyster_larvae/single" # + "/image_003.png"
    # path = "./dataset/oyster_larvae/grouped" + "/_image_003.png"
    # path = "./dataset/other_planktons/single"

    images = loadImages(path)


    counter = 000

    for image in images:
        # original = image       
        image = resize(image, 1280,720) 
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # image = preProcess(image)
        # cv2.imshow("IMG", image)
        
        # find_blob_centers(image)


        # cv2.imshow('Segmented Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # findBlobs(image)

        # findCentroids(image)
        # detect_and_crop_blobs(image)
        # cv2.waitKey(0)
        # barycenter_x, barycenter_y = compute_barycenter(image)

        # print("Barycenter coordinates (x, y):", barycenter_x, barycenter_y)
        equalized_image = cv2.equalizeHist(image)
        
        reference_image = cv2.imread('./ref.png')
        reference_image = resize(reference_image, 200, 200)

        correlationMap = sliding_correlation(reference_image, image)

        
        image = cv2.cvtColor(equalized_image, cv2.COLOR_BGR2GRAY)
        # reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

        normalized_image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        normalized_corr = cv2.normalize(correlationMap.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        


        # cropped = crop_around_barycenter(normalized_image, (500, 500))

        

        normalized_corr[normalized_corr < 0.9] = 0

        final = resize(normalized_corr)
        cv2.imshow("d",equalized_image)
        cv2.imshow("ddd",normalized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite("./normalDataset/planktons/image"+ str(counter) +".png", final)
        # counter+=1

        # cv2.imshow("IMG", final)
        # cv2.waitKey(0)
        # findCircles(image, 30, 500)

        # circles_info = extractCirclesInfo(image,50, 70 )



    