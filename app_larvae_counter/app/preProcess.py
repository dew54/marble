import numpy as np
import cv2
class PreProcess():
    def __init__():
        return



    @staticmethod
    def resize(image, w = 100, h = 100):
        final_image = cv2.resize(image, (w, h))
        return final_image

    @staticmethod
    def sliding_correlation(target, reference_path="ref.png"):
        
        reference = cv2.imread(reference_path)
        # target = cv2.imread("./ref.py")
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

    @staticmethod
    def findCenters(image):

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate the center of each contour
        center_coordinates = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cX = int(M['m10'] / M['m00'])
                cY = int(M['m01'] / M['m00'])
                center_coordinates.append((cX, cY))

        return center_coordinates

    @staticmethod
    def unsharpMask(gray_image):
        # Apply Gaussian blur to the grayscale image
        blurred_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Calculate the unsharp mask by subtracting the blurred image from the original
        unsharp_mask = cv2.subtract(gray_image, blurred_image)

        # Add the unsharp mask to the original image to enhance details
        return cv2.add(gray_image, unsharp_mask)

    @staticmethod
    def equalize(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Create a CLAHE object and apply CLAHE to the grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def normalize_data(array):
        mean = np.mean(array)
        variance = np.var(array)

        # Normalize the array using mean and variance
        normalized_array = (array - mean) / np.sqrt(variance)
        
        return normalized_array