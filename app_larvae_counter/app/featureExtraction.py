import cv2
from .preProcess import PreProcess
class FeatureExtraction():

    def __init__():
        return

    @staticmethod
    def extract_DESC(image, circle_info):
        

        image_descriptors = ()
        for info in circle_info:
            image_with_sift = image
            
            center_x, center_y = info  #info['Center']#map(int, info['Center'].strip('()').split(','))  # Convert the comma-separated string to tuple
            radius = 80#int(info['Radius'])

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


                    
                    # save_image_cv2(roi, "/dataset/oyster_larvae/singled", )
                    
                    # Extract SIFT keypoints and descriptors for the ROI
                    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                    gray_roi = PreProcess.unsharpMask(gray_roi)

                    # cv2.imshow("dd",gray_roi)
                    

                    gray_roi = cv2.equalizeHist(gray_roi)

                    # cv2.imshow("rr",gray_roi)
                    # cv2.waitKey(0)

                    

                    
                    # sift = cv2.SIFT_create()
                    # keypoints, descriptors = sift.detectAndCompute(gray_roi, None)
                    
                    orb = cv2.ORB_create()

                    keypoints, descriptors = orb.detectAndCompute(gray_roi, None)

                    keypoints_and_descriptors = list(zip(keypoints, descriptors))

                    keypoints_and_descriptors.sort(key=lambda kp_desc: kp_desc[0].response, reverse=True)

                    # Select the top N keypoints and descriptors
                    num_selected_keypoints = 49
                    selected_keypoints, selected_descriptors = zip(*keypoints_and_descriptors[:num_selected_keypoints])

                    # Convert selected_descriptors to a numpy array
                    descriptors = list(selected_descriptors)



                    if descriptors is not None:
                        # print(len(descriptors))

                        for desc in descriptors:

                            img_desc.append(desc)

                        image_descriptors += (img_desc,)
                    else:
                        print("No desc")

        return image_descriptors

