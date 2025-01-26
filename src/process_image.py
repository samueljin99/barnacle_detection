import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


def resize_raw_image(input_dir, image_file, top_left, top_right, bottom_left, bottom_right, processed_dir, resize_width=1536):
    """
    extract the area in focus from the given image, and resize to the desired dimension, save the result.
    """
    input_image_path = f"{input_dir}/{image_file}"  # Replace with your file path
    output_image_path = f"{processed_dir}/resized_{image_file}"  # Output file path

    # Define the four corner points of the polygon (replace with your values)
    polygon_points = np.array([
        top_left,  # Top-left corner
        top_right,  # Top-right corner
        bottom_right,  # Bottom-right corner
        bottom_left   # Bottom-left corner
    ], dtype=np.float32)

    # Define the destination points for the rectangle
    rectangle_points = np.array([
        [0, 0],  # Top-left corner
        [resize_width, 0],  # Top-right corner
        [resize_width, resize_width],  # Bottom-right corner
        [0, resize_width]  # Bottom-left corner
    ], dtype=np.float32)

    # Read the image
    image = cv2.imread(input_image_path)

    # Compute the perspective transformation matrix
    M = cv2.getPerspectiveTransform(polygon_points, rectangle_points)

    # Warp the image to the rectangle
    warped_image = cv2.warpPerspective(image, M, (resize_width, resize_width))

    # Save the resulting image
    cv2.imwrite(output_image_path, warped_image)

    print(f"Image warped and resized successfully. Saved as {output_image_path}.")


def create_image_tiles(source_dir, img_file, mask_file, grid_size=512):
    """
    make grids for the image, create N x N grids for the input image
    divide the image_file and the mask_file into smaller images. save them in processed_dir
    image_file and mask_file are of the same dimension.
    Each image should be of grid_size.
    The image should be named sequentially from 00000 to 00008 if it's 3x3 grid.

    Return the pair of image/mask grid file names, and image_index that can be used next.
    starting_sequence=1, padded_length=5
    """
    image = cv2.imread(f'{source_dir}/{img_file}.png')
    mask = cv2.imread(f'{source_dir}/{mask_file}.png')
    height, width, _ = image.shape

    # Calculate the size of each grid cell
    grids_height = height // grid_size
    grids_width = width // grid_size

    # Create a list to store the grid images
    sub_images = []

    # Loop through each grid cell and save the corresponding image
    for row in range(grids_height):
        for col in range(grids_width):
            # Define the start and end coordinates for each cell
            y_start = row * grid_size
            y_end = (row + 1) * grid_size
            x_start = col * grid_size
            x_end = (col + 1) * grid_size

            # Crop the image
            cropped_image = image[y_start:y_end, x_start:x_end]
            cropped_mask = mask[y_start:y_end, x_start:x_end]

            # Add to the grid images list
            # grid_images.append(cropped_image)
            sub_images.append((cropped_image, cropped_mask))

    return sub_images


def sample_random_subimages(num_samples, source_dir, img_file, mask_file, grid_size=512, seed=None):
    """
    generate samples from the image and mask file, each one subimage is grid_size
    :return: a list of (image, mask)
    """
    image = cv2.imread(f'{source_dir}/{img_file}.png')
    mask = cv2.imread(f'{source_dir}/{mask_file}.png')
    height, width, _ = image.shape

    height, width, _ = image.shape
    sub_images = []

    if seed is not None:
        random.seed(seed)

    for _ in range(num_samples):
        # Randomly choose the top-left corner of the sub-image
        top = random.randint(0, height - grid_size)
        left = random.randint(0, width - grid_size)

        # Extract the sub-image
        sub_image = image[top:top + grid_size, left:left + grid_size]
        sub_mask = mask[top:top + grid_size, left:left + grid_size]
        sub_images.append((sub_image, sub_mask))

    return sub_images


def flip_image(image, mode):
    """
    Flips an image based on the given mode.

    Args:
        image (numpy.ndarray): The input image.
        mode (int): Flip mode (0: vertical, 1: horizontal, -1: both).

    Returns:
        numpy.ndarray: The flipped image.
    """
    return cv2.flip(image, mode)


def rotate_image(image, angle):
    """
    Rotates an image by a given angle.

    Args:
        image (numpy.ndarray): The input image.
        angle (float): The angle by which to rotate the image, in degrees.

    Returns:
        numpy.ndarray: The rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated


def create_all_subimages(source_dir, target_dir, img_file, mask_file, grid_size, num_samples, starting_sequence=1, padded_length=5,
                         sample_seed=123):
    """
    create all combination of images, using tiling, sample subimages, flipping and rotation
    :param source_dir: where is the source images/masks
    :param target_dir: where to write the final images/masks
    :param img_file: name of the raw image file, do not include the .png part
    :param mask_file: name of the original mask file, size same as the img_file
    :param grid_size: size of the subimage
    :param starting_sequence:
    :param padded_length:
    :return:
    """
    # create tile subimages
    tile_subimages = create_image_tiles(source_dir, img_file, mask_file, grid_size)
    # sample addiitonal subimages
    sampling_size = num_samples - len(tile_subimages)
    sampled_subimages = sample_random_subimages(num_samples=sampling_size, source_dir=source_dir,
                                                img_file=img_file, mask_file=mask_file, grid_size=grid_size,
                                                seed=sample_seed)

    all_images = []
    for img, mask_img in tile_subimages + sampled_subimages:
        img_variations = generate_image_variations(img)
        mask_img_variations = generate_image_variations(mask_img)
        all_images.extend(zip(img_variations, mask_img_variations)) # each element in list is a pair of img and mask

    # write all images

    sub_image_files = []
    sub_mask_files = []

    image_index = starting_sequence
    for img, mask_img in all_images:
        image_sequence_str = pad_number_with_zeros(image_index, padded_length)

        sub_image_file = f'{img_file}_{image_sequence_str}.jpg'
        img_grid_file = f'{target_dir}/{sub_image_file}'
        sub_image_files.append(sub_image_file)

        sub_mask_file = f'{mask_file}_{image_sequence_str}.jpg'
        mask_grid_file = f'{target_dir}/{sub_mask_file}'
        sub_mask_files.append(sub_mask_file)

        cv2.imwrite(img_grid_file, img)
        cv2.imwrite(mask_grid_file, mask_img)
        image_index+=1

    return pd.DataFrame({"image": sub_image_files, "image_mask": sub_mask_files})


def generate_image_variations(image):
    """
    Generate all 8 unique variations of an image (rotations and flips).

    Args:
        image (numpy.ndarray): Input image (loaded using cv2.imread).
        mask_image

    Returns:
        list of (image, mask_image)
    """

    # Original image
    rotated_list = [image]
    # Rotate 90, 180, 270 degrees
    for rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        rotated = cv2.rotate(image, rotation)
        rotated_list.append(rotated)

    variations = []
    for img in rotated_list:
        flipped = cv2.flip(img, 1)  # horizontal flip
        variations.append(img)
        variations.append(flipped)

    return variations




def pad_number_with_zeros(number, padded_length):
    """
    Pads a number with leading zeros to ensure the total string length.
    """
    return f"{number:0{padded_length}}"


def get_bounding_boxes(image_path: str, bbox_min_len=12, is_show=False, is_verbose=False):
    """
    Get image width, height, and a list of bounding boxes for the given mask.
    The bounding boxes are (x, y, w, h), and it must be big enough on both sides (controled by bbox_min_len)
    """
    image = cv2.imread(image_path)
    width, height, _ = image.shape

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the image
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around each contour
    output_image = image.copy()
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < bbox_min_len or h < bbox_min_len:
            continue
        bounding_boxes.append((x, y, w, h))
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if is_show:
        # Display the result
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Bounding Boxes")
        plt.show()

    # Output the bounding boxes
    if is_verbose:
        for box in bounding_boxes:
            print(box)

    return width, height, bounding_boxes


from tqdm import tqdm
import os
import shutil as sh

def set_up_training_file(data_dir, df, source_data_dir):
    """
    set up file directories for training according to Yolo set up.
    :param data_dir: the destination data dir
    :param df: what images has what bounding boxes, for what split, etc
    :param source_data_dir: where the source images are
    :return:
    """
    df = df.copy()

    # recreate the data dir
    if os.path.exists(data_dir):
        sh.rmtree(data_dir)
        os.makedirs(data_dir)

    # convert to ratio values
    df['x_center'] = df['x_center']/df['image_width']
    df['y_center'] = df['y_center']/df['image_height']
    df['w'] = df['w']/df['image_width']
    df['h'] = df['h']/df['image_height']

    # loop through the bounding boxes per image
    for (image_name, training_split), mini in tqdm(df.groupby(['image', 'split'])):
        name, _ = image_name.split('.')
        # where to save the files
        # storage path for labels
        if not os.path.exists(f'{data_dir}/labels/{training_split}'):
            os.makedirs(f'{data_dir}/labels/{training_split}')
        with open(f'{data_dir}/labels/{training_split}/{name}.txt', 'w+') as f:
            # normalize the coordinates in accordance with the Yolo format requirements
            row = mini[['classes','x_center','y_center','w','h']].astype(float).values
            row = row.astype(str)
            for j in range(len(row)):
                text = ' '.join(row[j])
                f.write(text)
                f.write("\n")
        if not os.path.exists(f'{data_dir}/images/{training_split}'):
            os.makedirs(f'{data_dir}/images/{training_split}')
        # no preprocessing needed for images => copy them as a batch
        sh.copy(f"{source_data_dir}/{image_name}",f'{data_dir}/images/{training_split}/{image_name}')

    return df


def show_images_with_bbox(image_path, bbox_file, thickness=2):
    """
    Show images with bounding box overlay
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Get the dimensions of the image
    height, width, _ = image.shape

    # Read the bounding box file
    if bbox_file:
        with open(bbox_file, 'r') as file:
            lines = file.readlines()
    else:
        lines = []

    # Process each bounding box
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue  # Skip invalid lines

        # Extract bounding box coordinates and scale to image size
        x_center = float(parts[1]) * width
        y_center = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the bounding box on the image
        color = (0, 0, 255)  # Green color for the bounding box
        thickness = 2  # Thickness of the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Bounding Boxes")
    plt.show()


def show_bounding_boxes(image_path, results):
    """
    Display the original image with bounding boxes overlaid.

    Args:
        image_path (str): Path to the original image.
        results: YOLO model inference results (e.g., from `model.predict()`).
    """
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display in Matplotlib

    # Loop through all detections in the results
    for result in results:
        for box in result.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])       # Class ID

            # Draw the bounding box on the image
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
            label = f"Class {class_id}: {confidence:.2f}"
            # cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()