import cv2

from main import test_find_tfl_lights

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


except ImportError:
    print("Need to fix the installation")
    raise



def create_labels(red_x, red_y, green_x, green_y):
    pass


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r"C:\Users\naorb\Desktop\Scale-up\projects\mobileye\leftImg8bit\leftImg8bit\train"

    # for filename in os.listdir(args.dir):
    #     flist = glob.glob(os.path.join(args.dir, filename, '*_leftImg8bit.png'))
    #     for image in flist:
    #         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
    #
    #         if not os.path.exists(json_fn):
    #             json_fn = None
    #         test_find_tfl_lights(image, json_fn)
    counter = 1
    flist = glob.glob(os.path.join(default_base, "aachen", '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png').replace('leftImg8bit', 'gtFine', 2)
        print(counter)
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)


        counter += 1

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)

loaded_arr = np.fromfile("data.bin",  dtype=np.uint8)
load_original_arr = loaded_arr.reshape(160, 80, 3)
plt.imshow(load_original_arr)
plt.show()
print("fool")
# if __name__ == '__main__':
#     main()