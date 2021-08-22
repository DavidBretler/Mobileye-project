import heapq

import cv2
from matplotlib.testing.compare import compare_images
from numpy import amax
# from sympy.stats.frv_types import scipy
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


def stam(image):

    # plt.imshow(image)
    # plt.show()
    # split the image into its BGR components
    mask_red = np.all(image[:,:] <= [180, 235, 235], axis=-1)
    mask_green = np.all(image[:,:] <= [245, 237, 245], axis=-1)
    mask_blue = np.all(image[:,:] <= [245, 245, 190], axis=-1)

    mask_white = np.ma.mask_or(np.all(image[:, :] >= [200, 200, 200], axis=-1), np.all(image[:, :] <= [195, 195, 195], axis=-1))
    mask = np.ma.mask_or(np.ma.mask_or(mask_red, mask_green), np.ma.mask_or(mask_blue, mask_white))
    image[mask] = [0, 0, 0]



    image[np.logical_not(mask)] = [255, 255, 255]

    # plt.imshow(image)
    # plt.show()
    blurred = cv2.GaussianBlur(image, (17, 17), 0)
    fixed = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)[1]

    # plt.imshow(fixed)
    # plt.show()
    # (B, G, R) = cv2.split(image)
    # # find the maximum pixel intensity values for each
    # # (x, y)-coordinate,, then set all pixel values less
    # # than M to zero
    #
    # # R = cv2.threshold(R, 90, 255, cv2.THRESH_BINARY)[1]
    # # G = cv2.threshold(G, 90, 255, cv2.THRESH_BINARY)[1]
    # # B = cv2.threshold(B, 90, 255, cv2.THRESH_BINARY)[1]
    #
    # M = np.maximum(np.maximum(R, G), B)
    #
    #
    #
    # R[R < M] = 0
    # G[G < M] = 0
    # B[B < M] = 0
    # # merge the channels back together and return the image
    # final = cv2.merge([B, G, R])

    return fixed


def red_detection(img):
    epsilon = 10
    img_new = img.copy()
    for j in range(len(img_new)):
        for i in range(len(img_new[j])):
            if img[j][i][0] > img[j][i][1] + epsilon and img[j][i][0] > img[j][i][2] + epsilon:
                img_new[j][i] = [255, 255, 255]
            else:
                img_new[j][i] = [0, 0, 0]
    return img_new


def check_red(pixel_colors):
    epsilon = 10
    return pixel_colors[0] > pixel_colors[1] + epsilon and pixel_colors[0] > pixel_colors[2] + epsilon


# paklocalmax
def n_max(ary, n):
    lst = []
    print(lst)
    for i in range(len(ary)):
        for j in range(len(ary[i])):
            if len(lst) < n:
                lst.append((ary[i][j], i, j))
                lst = sorted(lst, key=lambda x: x[0])
            else:
                # [[1,4,3,6], [3,4,1,7], [9,1,1]]
                if ary[i][j] > ary[0][0]:
                    lst[0] = (ary[i][j], i, j)
                    lst = sorted(lst, key=lambda x: x[0])
    return lst


def check_red_green(item):
    if (item[0] > 240) and (item[2] > 240) and (item[1] > 240):
        return True
    return False
    # if (item[2] > 200) and ((item[0] < 130) or (item[1] < 130)):
    #     return True
    # return False


def find_tfl_lights(c_image: np.ndarray, cv2=None, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    # n = 10
    # lst = []
    #             if len(lst) < n:
    #                 lst.append((max_filter_image[i][j], i, j))
    #                 lst = sorted(lst, key=lambda x: x[0])
    #             else:
    #                 # [[1,4,3,6], [3,4,1,7], [9,1,1]]
    #                 if max_filter_image[i][j] > lst[0][0]:
    #                     lst[0] = (max_filter_image[i][j], i, j)
    #                     lst = sorted(lst, key=lambda x: x[0])
    #
    # # print(n_max(np.array([[1,4,3,6], [3,4,1,7], [9,1,1]]), 3))
    # lst_index = [lst[2] for x in lst]
    # lst_indey = [lst[1] for x in lst]
    # lst = n_max(lst_index2, 3)
    # # plt.imshow(x)
    # plt.show()



    gray_image = np.dot(c_image[..., :3], [0.2125, 0.7154, 0.0721])


    # gray_image = c_image.copy()[:,:,0]
    # plt.imshow(gray_image)
    # plt.show()

    kernel = np.array(Image.open('k3.png').convert('L'))
    kernel = kernel.astype('f')
    offset = kernel.sum() / (len(kernel[0]) * len(kernel))

    for j in range(len(gray_image)):
        for i in range(len(gray_image[j])):
            # cou = cou + kernel[j][i]
            gray_image[j][i] = float(gray_image[j][i]) - offset

    for j in range(len(kernel)):
        for i in range(len(kernel[j])):
            kernel[j][i] = float(kernel[j][i]) - offset

    # fixed_image = sg.convolve(gray_image, kernel, mode='same', method='auto')
    # mask = stam(c_image.copy())
    # fixed_image[np.logical_not(mask)] += 900
    # fixed_image[mask] -= 90

    fixed_image = sg.convolve(np.dot(stam(c_image.copy())[..., :3], [0.2125, 0.7154, 0.0721]), kernel, mode='same', method='auto')

    # plt.imshow(fixed_image)
    # plt.show()


    ax = plt.subplot(2, 1, 1)
    ax.set_title("Before")
    plt.imshow(fixed_image)

    max_filter_image = scipy.ndimage.maximum_filter(fixed_image, 50)
    # plt.imshow(max_filter_image)
    # plt.show()
    lst_index1 = []
    lst_index2 = []

    for i in range(len(max_filter_image)):
        for j in range(len(max_filter_image[i])):
            if max_filter_image[i][j] == fixed_image[i][j] and max_filter_image[i][
                j] > 1500000:  # and check_red(c_image[i][j]):#(np.mean(max_filter_image)*5):
                lst_index1.append(i)
                lst_index2.append(j)

    ### USE HELPER FUNCTIONS ###
    return lst_index2, lst_index1, lst_index2, lst_index1


### GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
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


def test_find_tfl_lights(image_path, label_path=None, fig_num=None):
    """
    Run the attention code
    """


    image = np.array(Image.open(image_path))
    objects = None
    if label_path is None:
        image_labels = None
    else:
        image_labels = np.array(Image.open(label_path))

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    tfl_images = []
    no_tfl_images = []

    zeroes = np.zeros((len(image)+80, len(image[0])+80, 3))
    zeroes[40:image.shape[0]+40, 40:image.shape[1]+40] = image
    padded_image = zeroes.astype(dtype=np.uint8)

    for x, y in zip(red_x, red_y):
        if image_labels[y][x] == 19:
                cropped = padded_image[y:y+80, x:x+80, :]
                tfl_images.append(cropped)
        else:
                cropped = padded_image[y:y+80, x:x+80, :]
                no_tfl_images.append(cropped)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)
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
    test_find_tfl_lights('cologne_000113_000019_leftImg8bit.png', 'cologne_000113_000019_gtFine_labelIds.png')
    plt.show(block=True)
