<<<<<<< HEAD
import cv2
import scipy

try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image, ImageEnhance, ImageFilter
    import matplotlib.pyplot as plt
    import matplotlib

except ImportError:
    print("Need to fix the installation")
    raise

data_path = "data.bin"
labels_path = "labels.bin"


def highlight_lights(image):
    mask_red = np.all(image[:, :] <= [180, 235, 235], axis=-1)
    mask_green = np.all(image[:, :] <= [245, 235, 245], axis=-1)
    mask_blue = np.all(image[:, :] <= [245, 245, 190], axis=-1)

    mask_white = np.ma.mask_or(np.all(image[:, :] >= [190, 190, 190], axis=-1),
                               np.all(image[:, :] <= [155, 155, 155], axis=-1))

    mask = np.ma.mask_or(np.ma.mask_or(mask_red, mask_green), np.ma.mask_or(mask_blue, mask_white))
    image[mask] = [0, 0, 0]

    image[np.logical_not(mask)] = [255, 255, 255]

    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (7, 7), 0)
    fixed = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    # another way to detect the lights (get tfl less = noise less)
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    # blurred = cv2.GaussianBlur(fixed, (17, 17), 0)
    # fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    return fixed[..., :3]


def find_tfl_lights(c_image: np.ndarray, cv2=None, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = np.array(Image.open('kernel.png').convert('L'))
    kernel = kernel.astype('f')

    highlighted_image = highlight_lights(c_image.copy())
    fixed_image = \
        sg.convolve(np.dot(highlighted_image, [0.2125, 0.7154, 0.0721]), kernel, mode='same', method='auto')

    max_filter_image = scipy.ndimage.maximum_filter(fixed_image, 50)

    red_index1 = []
    red_index2 = []
    green_index1 = []
    green_index2 = []

    for i in range(len(max_filter_image)):
        for j in range(len(max_filter_image[i])):
            if max_filter_image[i][j] == fixed_image[i][j] and max_filter_image[i][j] > 1500000:
                if c_image[i][j][0] > c_image[i][j][1] >= c_image[i][j][2]:
                    red_index1.append(i)
                    red_index2.append(j)
                else:
                    green_index1.append(i)
                    green_index2.append(j)
    return red_index2, red_index1, green_index2, green_index1


def show_image_and_gt(image, objs, fig_num=None):
    # plt.figure(fig_num).clf()
    # plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            # plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def show_find_tfl_lights(image_path):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)
    plt.imshow(image)
    plt.show()


def test_find_tfl_lights(image_path, label_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    image_labels = np.array(Image.open(label_path))

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    tfl_images = []
    no_tfl_images = []

    zeroes = np.zeros((len(image) + 81, len(image[0]) + 81, 3))
    zeroes[41:image.shape[0] + 41, 41:image.shape[1] + 41] = image
    padded_image = zeroes.astype(dtype=np.uint8)

    for x1, y1, x2, y2 in zip(red_x, red_y, green_x, green_y):
        if image_labels[y1][x1] == 19:
            cropped = padded_image[y1:y1 + 81, x1:x1 + 81, :]
            tfl_images.append(cropped)
        else:
            cropped = padded_image[y1:y1 + 81, x1:x1 + 81, :]
            no_tfl_images.append(cropped)

        if image_labels[y2][x2] == 19:
            cropped = padded_image[y2:y2 + 81, x2:x2 + 81, :]
            tfl_images.append(cropped)
        else:
            cropped = padded_image[y2:y2 + 81, x2:x2 + 81, :]
            no_tfl_images.append(cropped)

    return tfl_images, no_tfl_images


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help="Path to Image")
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = "INSERT_YOUR_DIR_WITH_PNG_AND_JSON_HERE"

    # image = np.array(Image.open(r'C:\Users\elyasaf\PycharmProjects\MobilayProject\test3.jpg'))
    # find_tfl_lights(image)
    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
    test_find_tfl_lights('img3.png')

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    # main()
    show_find_tfl_lights('gallery/img2.png')  # , 'cologne_000113_000019_gtFine_labelIds.png')
    plt.show(block=True)
=======
try:
    import os
    import json
    import glob
    import argparse
    import numpy as np
    import cv2
    import scipy
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter
    from PIL import Image, ImageEnhance, ImageFilter
    import matplotlib.pyplot as plt
    import matplotlib

except ImportError:
    print("Need to fix the installation")
    raise


def highlight_lights(image: np.ndarray):
    """
    turning an numpy array image into a black white image with white spots
    which are suspicious for being a traffic light, using masks, cv2 functions as blur and threshold
    :param image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 2D black-white image with black background and white spots which are the tfl that detected
    """
    mask_red = np.all(image[:, :] <= [180, 235, 235], axis=-1)
    mask_green = np.all(image[:, :] <= [245, 235, 245], axis=-1)
    mask_blue = np.all(image[:, :] <= [245, 245, 190], axis=-1)

    mask_white = np.ma.mask_or(np.all(image[:, :] >= [190, 190, 190], axis=-1),
                               np.all(image[:, :] <= [155, 155, 155], axis=-1))

    mask = np.ma.mask_or(np.ma.mask_or(mask_red, mask_green), np.ma.mask_or(mask_blue, mask_white))
    image[mask] = [0, 0, 0]

    image[np.logical_not(mask)] = [255, 255, 255]

    # a better/worse way to detect the lights (get more tfl results = get more noises)
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (7, 7), 0)
    fixed = cv2.threshold(blurred, 55, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]
    blurred = cv2.GaussianBlur(fixed, (19, 19), 0)
    fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    # another way to detect the lights (get less tfl results = get less noises)
    # blurred = cv2.GaussianBlur(image, (3, 3), 0)
    # fixed = cv2.threshold(blurred, 26, 255, cv2.THRESH_BINARY)[1]
    # blurred = cv2.GaussianBlur(fixed, (17, 17), 0)
    # fixed = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)[1]

    return fixed[..., :3]  # return result 2D


def find_tfl_lights(c_image: np.ndarray):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel = np.array(Image.open('kernel.png').convert('L'))
    kernel = kernel.astype('f')

    highlighted_image = highlight_lights(c_image.copy())
    fixed_image = \
        sg.convolve(np.dot(highlighted_image, [0.2125, 0.7154, 0.0721]), kernel, mode='same', method='auto')

    max_filter_image = scipy.ndimage.maximum_filter(fixed_image, 50)

    red_index1 = []
    red_index2 = []
    green_index1 = []
    green_index2 = []

    for i in range(len(max_filter_image)):
        for j in range(len(max_filter_image[i])):
            if max_filter_image[i][j] == fixed_image[i][j] and max_filter_image[i][j] > 1500000:
                if c_image[i][j][0] > c_image[i][j][1] >= c_image[i][j][2]:
                    red_index1.append(i)
                    red_index2.append(j)
                else:
                    green_index1.append(i)
                    green_index2.append(j)
    return red_index2, red_index1, green_index2, green_index1


def show_find_tfl_lights(image_path):
    """
     function that gets an image path and detect on here the suspicious places for being a TFL
     the function visually shows with matplotlib the TFL's detected
    :param image_path:  path to the original image which we want to detect on it the TFL's
    :return: None
    """
    image = np.array(Image.open(image_path))
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)
    plt.imshow(image)
    plt.show()
    print("You should now see some images, with the ground truth marked on them. Close all to quit.")


def test_find_tfl_lights(image_path, label_path=None):
    """
    function that gets an image path and labeled image path
    and activate the detect_lights function, after that gets the x and y results as lists
    then compare the location of the (x,y) tfl, with the exact location at the labeled_image,
    if the value equals to 19, then it means that its a traffic light, so add it to tfl_list,
    else - add it to no_tfl_list.
    :param image_path: path to the original image which we want to detect on it the TFL's
    :param label_path: path to the labeled image which we want to compare the id of pixel if its belong to TFL
     (Ground Truth)
    :return: two lists of images with shape 81x81x3 represents TFL's and no TFL's
    """
    image = np.array(Image.open(image_path))
    image_labels = np.array(Image.open(label_path))

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    tfl_images = []
    no_tfl_images = []

    # padding with 40 pixels
    zeroes = np.zeros((len(image) + 81, len(image[0]) + 81, 3))
    zeroes[41:image.shape[0] + 41, 41:image.shape[1] + 41] = image
    padded_image = zeroes.astype(dtype=np.uint8)
    # there's no difference between green or red colors of lights in the TFL
    for x1, y1, x2, y2 in zip(red_x, red_y, green_x, green_y):
        if image_labels[y1][x1] == 19:
            cropped = padded_image[y1:y1 + 81, x1:x1 + 81, :]
            tfl_images.append(cropped)
        else:
            cropped = padded_image[y1:y1 + 81, x1:x1 + 81, :]
            no_tfl_images.append(cropped)
        if image_labels[y2][x2] == 19:
            cropped = padded_image[y2:y2 + 81, x2:x2 + 81, :]
            tfl_images.append(cropped)
        else:
            cropped = padded_image[y2:y2 + 81, x2:x2 + 81, :]
            no_tfl_images.append(cropped)

    return tfl_images, no_tfl_images


if __name__ == '__main__':
    """
    simple way to run code is running this file,
    choose the image which you want to detect the traffic lights from
    """
    show_find_tfl_lights('gallery/img2.png')  # , 'cologne_000113_000019_gtFine_labelIds.png')
    plt.show(block=True)
>>>>>>> 9e06529581f3d1e30aa356a7dca92a74e576027d
