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

    # tfl_list = []
    # no_tfl_list = []
    # flist = glob.glob(os.path.join(default_base, "aachen", '*_leftImg8bit.png'))
    # for image in flist:
    #     label_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png').replace('leftImg8bit', 'gtFine', 2)
    #     print(counter)
    #     if not os.path.exists(label_fn):
    #         label_fn = None
    #     temp_tfl_list, temp_no_tfl_list = test_find_tfl_lights(image, label_fn)
    #     tfl_list += temp_tfl_list
    #     no_tfl_list += temp_no_tfl_list
    #     counter += 1

    default_base_train = r"C:\Users\naorb\Desktop\Scale-up\projects\mobileye\leftImg8bit\leftImg8bit\train"
    default_base_val = r"C:\Users\naorb\Desktop\Scale-up\projects\mobileye\leftImg8bit\leftImg8bit\val"
    data_dir_train = r'gtFine2\train'
    data_dir_val = r'gtFine2\val'

    counter_result = write_dataset(default_base_train, data_dir_train, 2)
    write_dataset(default_base_val, data_dir_val, int(counter_result*(1/6)))
    print("You should now see some images, with the ground truth marked on them. Close all to quit.")


def balanced_quantity(tfl_images, no_tfl_images):
    count = 0
    mirrored_tfl = []
    mirrored_no_tfl = []
    total_len = (len(tfl_images) + len(no_tfl_images)) * 0.4

    while len(no_tfl_images) + len(mirrored_no_tfl) < total_len and count < len(no_tfl_images):
        mirrored_no_tfl.append(np.fliplr(no_tfl_images[count]))
        count += 1

    while len(tfl_images) + len(mirrored_tfl) < total_len and count < len(tfl_images):
        mirrored_tfl.append(np.fliplr(tfl_images[count]))
        count += 1

    tfl_images += mirrored_tfl
    no_tfl_images += mirrored_no_tfl

    while len(tfl_images) < (len(tfl_images) + len(no_tfl_images)) * 0.4:
        no_tfl_images.pop()

    while len(no_tfl_images) < (len(tfl_images) + len(no_tfl_images)) * 0.4:
        tfl_images.pop()

    return tfl_images, no_tfl_images


def write_dataset(default_base: str, data_dir: str, max_counter: int):
    counter = 1
    tfl_list = []
    no_tfl_list = []
    for filename in os.listdir(default_base):
        print(filename)
        flist = glob.glob(os.path.join(default_base, filename, '*_leftImg8bit.png'))
        for image in flist:
            label_fn = image.replace('_leftImg8bit.png', '_gtFine_labelIds.png').replace('leftImg8bit', 'gtFine', 2)
            print(counter)
            if not os.path.exists(label_fn):
                continue
            temp_tfl_list, temp_no_tfl_list = test_find_tfl_lights(image, label_fn)
            tfl_list += temp_tfl_list
            no_tfl_list += temp_no_tfl_list
            counter += 1

            if counter > max_counter:
                break
        if counter > max_counter:
            break
    tfl_list, no_tfl_list = balanced_quantity(tfl_list, no_tfl_list)
    labels = [1]*len(tfl_list) + [0]*len(no_tfl_list)
    data = tfl_list + no_tfl_list
    data = np.array(data, dtype=np.uint8)
    data.tofile(os.path.join(data_dir, 'data.bin'))
    labels = np.array(labels, dtype=np.uint8)
    labels.tofile(os.path.join(data_dir, 'labels.bin'))
    return counter


# loaded_data = np.fromfile("gtFine/val/data.bin",  dtype=np.uint8)
# loaded_labels = np.fromfile("gtFine/train/labels.bin",  dtype=np.uint8)
# a = loaded_data.reshape(len(loaded_data)//19683, 81, 81, 3)
#
# for img in a:
#     plt.imshow(img)
#     plt.show()

#
if __name__ == '__main__':
    main()