import cv2; 
import matplotlib.pyplot as plt
import math
import numpy as np



# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    return cv2.imread(image_path)
    

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    img_gray = image.copy()
    for i in range(img_gray.shape[0]):
        for j in range(img_gray.shape[1]):
            img_gray[i, j, :] = [0.299*img_gray[i, j, 2] + 0.587*img_gray[i, j, 1] + 0.114*img_gray[i, j, 0]]
    return img_gray


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    return cv2.flip(image, 1)


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    """
    image_float = image.astype(np.float32)

    # Calculate the rotation matrix
    height, width = image_float.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Rotate the image
    image_rotated = cv2.warpAffine(image_float, rotation_matrix, (width, height))

    # Convert the image back to uint8
    image_rotated = image_rotated.astype(np.uint8)
"""
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))
    


if __name__ == "__main__":
    # Load an image from file
    img = load_image("uet.png")

    # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "lena_gray_rotated.jpg")

    # Show the images
    plt.show() 
