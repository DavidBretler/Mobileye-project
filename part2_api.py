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
    default_base = r"C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2\leftImg8bit\test"

    if args.dir is None:
        args.dir = default_base

    # for filename in os.listdir(args.dir):
    #     flist = glob.glob(os.path.join(args.dir, filename, '*_leftImg8bit.png'))
    #     for image in flist:
    #         json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
    #
    #         if not os.path.exists(json_fn):
    #             json_fn = None
    #         test_find_tfl_lights(image, json_fn)
    counter = 1
    flist = glob.glob(os.path.join(args.dir, "berlin", '*_leftImg8bit.png'))
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
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


if __name__ == '__main__':
    main()