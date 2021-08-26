import pickle
import random

from PIL import Image

from View import view
import matplotlib._png as png
import numpy as np
from Model.model_based_tfl_detection import find_tfl_lights
import Model.neural_network_model
import Model.dataset_creation
play_list_path = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2\Data\Resource\play_list.pls'
PATH2 = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2'


class TFL_manager:
    focal = ''
    pp = ''
    Ego_motion = ''
    lst_red_lights = []
    lst_green_lights = []

    def __init__(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def run(self, prev_frame, curr_frame, EM,neural_network_model):
        self.Ego_motion = EM

        # part one
        red_lights, green_lights = find_tfl_lights(np.array(Image.open(PATH2+'\\'+curr_frame)))
        # print(len(lst_lights))

        # part two use neural network
        red_lights, green_lights = Model.neural_network_model.predict_and_evaluate\
            (neural_network_model,np.array(Image.open(PATH2+'\\'+curr_frame)))
        # print(len(lst_tfl))

        if prev_frame == 'not initialized':
            self.green_lights = green_lights
            self.red_lights = red_lights
            return red_lights, green_lights, []

        # part three
        lst_dist = self.calc_dist(red_lights, green_lights,'',
                                  TFL_manager.lst_red_lights, TFL_manager.lst_green_lights)

        self.green_lights = green_lights
        self.red_lights = red_lights

        return red_lights, green_lights, lst_dist

    # def find_lights(self, cur_frame):
    #     return find_tfl_lights(np.array(cur_frame))
    #     # res = []
    #     # res_color = []
    #     # for i in range(10):
    #     #     res.append([random.randint(1, 1000), random.randint(100, 1000)])
    #     #     res_color.append('RED')
    #     # return res, res_color
    #
    # def find_TFL(self, cur_frame, red_lights, green_lights):
    #     res_tfl = []
    #     res_color_tfl = []
    #     for i in range(len(red_lights)):
    #         if i % 3 == 0:
    #             res_tfl.append(red_lights[i])
    #     for i in range(len(green_lights)):
    #         if i % 3 == 0:
    #             res_color_tfl.append(green_lights[i])
    #     return red_lights, green_lights

    def calc_dist(self, red_lights, green_lights, lst_tfl_cur,
                  lst_colors_cur, lst_tfl_prev):
        res_dist = []
        for i in range(len(red_lights)+len(green_lights)):
            res_dist.append((i * 25.6 + i * i, i * 25.6 + i))
        return res_dist


def init():
    # create dataset for neural network
    Model.dataset_creation.main()

    # train neural network
    neural_network_model=Model.neural_network_model.init()

    pkl_path, frame_id, frame_list = get_frames()

    data = get_data(pkl_path)
    focal = data['flx']
    pp = data['principle_point']

    tfl_manager = TFL_manager(focal, pp)
    run(tfl_manager, frame_list, data, int(frame_id),neural_network_model)


def run(tfl_manager, frame_list, data, frame_id,neural_network_model):
    prev_frame = 'not initialized'
    EM = np.eye(4)
    for i in range(len(frame_list)):
        cur_frame = frame_list[i]
        if i != 0:
            prev_frame = frame_list[i - 1]
            EM = np.eye(4)
            for i in range(frame_id, frame_id + 1):
                EM = np.dot(data['egomotion_' + str(i - 1) + '-' + str(i)], EM)

        red_lights, green_lights, lst_dist = tfl_manager.run(prev_frame, cur_frame, EM,neural_network_model)
        if prev_frame != 'not initialized':
            view.visualize(cur_frame,  red_lights, green_lights, lst_dist, frame_id)
        frame_id += 1


def get_frames():
    global play_list_path
    with open(play_list_path, 'r') as file_object:
        paths = file_object.readlines()
        frames = [x[:-1] for x in paths[2:]]
    return paths[0][:-1], paths[1], frames


def get_data(pkl_path):
    pkl_path = PATH2 + '\\' + pkl_path
    with open(pkl_path, 'rb') as pklfile:
        data = pickle.load(pklfile, encoding='latin1')
    return data


init()
