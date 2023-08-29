import os
import cv2
import random
import numpy as np

def resize_image(image, target_size):
    return cv2.resize(image, target_size)

def rotate_image(image, angle):

    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))

    # Specify the radius of expansion
    expand_radius = 3  # Adjust this value as needed
    # Find the coordinates where the image is 0-valued
    zero_coords = np.where(rotated_image == 0)
    
    new_level = 200 
    # Create a binary mask from zero_coords
    mask = np.zeros_like(rotated_image, dtype=np.uint8)
    mask[zero_coords] = 255

    # Dilate the mask to expand the area
    kernel = np.ones((expand_radius * 2 + 1, expand_radius * 2 + 1), dtype=np.uint8)
    dilated_mask = cv2.dilate(mask, kernel)

    # Apply the dilated mask to the original image
    expanded_image = np.where(dilated_mask == 255, 0, rotated_image)

    zero_coords = np.where(expanded_image == 0)
    gray_roi = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2GRAY)



    new_level = max(gray_roi[50, 7] , gray_roi[50, -7])
    expanded_image[zero_coords] = new_level




    return expanded_image

def flip_image(image):
    return cv2.flip(image, 1)  # 1 for horizontal flip

def central_crop(image, crop_size):
    y1 = int((image.shape[0] - crop_size[0]) / 2)
    y2 = y1 + crop_size[0]
    x1 = int((image.shape[1] - crop_size[1]) / 2)
    x2 = x1 + crop_size[1]
    return image[y1:y2, x1:x2]

def augment_images(input_folder, output_folder, target_num_images, target_size=(100, 100), rotation_angles=[-30, -15, 0, 15, 30], flip=True, crop=True, brightness_range=(0.7, 1.3), contrast_range=(0.7, 1.3), crop_size=(100, 100)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    num_augmentations_per_image = max(1, target_num_images // len(os.listdir(input_folder)))

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Resize to a fixed size
            image = resize_image(image, target_size)

            for _ in range(num_augmentations_per_image):
                augmented_image = image.copy()

                # Rotation
                if rotation_angles:
                    angle = random.choice(rotation_angles)
                    augmented_image = rotate_image(augmented_image, angle)

                # Flipping
                if flip and random.random() > 0.5:
                    augmented_image = flip_image(augmented_image)

                # Central cropping
                if crop:
                    augmented_image = central_crop(augmented_image, crop_size)

                # Save augmented image
                output_filename = f"augmented_{filename[:-4]}_{_+1}.jpg"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, augmented_image)

if __name__ == "__main__":
    input_folder = "./normalDataset/larvae"
    output_folder = "./normalDataset/augmented/larvae"
    target_num_images = 2000  # Total number of images after augmentation
    rotation_angles = [-45, -30, -15, 0, 15, 30, 45]  # List of rotation angles to choose from
    augment_images(input_folder, output_folder, target_num_images, rotation_angles=rotation_angles)
