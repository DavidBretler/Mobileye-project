import pickle

from Model import SFM
from View import view
import matplotlib._png as png
import numpy as np
from Model.model_based_tfl_detection import find_tfl_lights
import Model.neural_network_model
import Model.dataset_creation

play_list_path = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2\Data\Resource\play_list.pls'
PATH2 = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2'


class FrameContainer(object):
    def __init__(self, img_path):
        self.img = png.read_png_int(img_path)
        self.traffic_light = []
        self.traffic_light_red = []
        self.traffic_light_green = []
        self.traffic_lights_3d_location = []
        self.EM = []
        self.corresponding_ind = []
        self.valid = []
        self.first_frame = False


class TFL_manager:
    focal = ''
    pp = ''
    Ego_motion = ''
    lst_red_lights = []
    lst_green_lights = []

    def __init__(self, focal, pp):
        self.focal = focal
        self.pp = pp

    def run(self, prev_container, cur_container, EM, neural_network_model):
        self.Ego_motion = EM

        # part one
        # red_lights, green_lights = find_tfl_lights(np.array(Image.open(PATH2 + '\\' + curr_frame)))
        red_lights, green_lights = find_tfl_lights(np.array(cur_container.img))

        # print(len(lst_lights))

        # part two use neural network
        # img_array = np.array(Image.open(PATH2 + '\\' + curr_frame))
        red_lights, green_lights = Model.neural_network_model.find_tfl(model=neural_network_model,
                                                                       image=np.array(cur_container.img)
                                                                       , red_lights=red_lights,
                                                                       green_lights=green_lights)
        # red_lights, green_lights = Model.neural_network_model.predict_and_evaluate \
        #     (neural_network_model, np.array(Image.open(PATH2 + '\\' + curr_frame)))
        # print(len(lst_tfl))
        cur_container.traffic_light_red = red_lights
        cur_container.traffic_light_green = green_lights
        cur_container.traffic_light=red_lights+green_lights

        if  cur_container.first_frame:
            self.green_lights = green_lights
            self.red_lights = red_lights
            return cur_container

        # part three
        # lst_dist = self.calc_dist(red_lights, green_lights, '',
        #                           TFL_manager.lst_red_lights, TFL_manager.lst_green_lights)
        curr_container = SFM.calc_TFL_dist(prev_container, cur_container, self.focal, self.pp)

        self.green_lights = green_lights
        self.red_lights = red_lights

        return curr_container

    # stab part one
    # def find_lights(self, cur_frame):
    #     return find_tfl_lights(np.array(cur_frame))
    #     # res = []
    #     # res_color = []
    #     # for i in range(10):
    #     #     res.append([random.randint(1, 1000), random.randint(100, 1000)])
    #     #     res_color.append('RED')
    #     # return res, res_color

    # stab part two
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

    # stab part three
    # def calc_dist(self, red_lights, green_lights, lst_tfl_cur,
    #               lst_colors_cur, lst_tfl_prev):
    #     res_dist = []
    #     for i in range(len(red_lights) + len(green_lights)):
    #         res_dist.append((i * 25.6 + i * i, i * 25.6 + i))
    #     return res_dist


def init():
    # create dataset for neural network from part one points
    Model.dataset_creation.main()

    # train neural network
    neural_network_model = Model.neural_network_model.init()

    pkl_path, frame_id, frame_list = get_frames()

    data = get_data(pkl_path)
    focal = data['flx']
    pp = data['principle_point']

    tfl_manager = TFL_manager(focal, pp)
    run(tfl_manager, frame_list, data, int(frame_id), neural_network_model)


def run(tfl_manager, frame_list, data, frame_id, neural_network_model):
    EM = np.eye(4)
    list_tfl = []
    for i in range(len(frame_list)):
        cur_container = FrameContainer(PATH2 + '\\' + frame_list[i])
        if i != 0:
            EM = np.eye(4)

            if i < len(frame_list):
                for j in range(frame_id, frame_id + 1):
                    EM = np.dot(data['egomotion_' + str(j - 1) + '-' + str(j)], EM)
            prev_container = FrameContainer(PATH2 + '\\' + frame_list[i - 1])
            prev_container.traffic_light_red = list_tfl[i-1][0]
            prev_container.traffic_light_green = list_tfl[i-1][1]
            prev_container.traffic_light=list_tfl[i-1][0]+list_tfl[i-1][1]

        else:
            prev_container = FrameContainer(PATH2 + '\\' + frame_list[i])
            cur_container.first_frame = True

        cur_container.EM = EM
        cur_container = tfl_manager.run(prev_container, cur_container, EM, neural_network_model)

        if not cur_container.first_frame:
            view.visualize(cur_container, frame_id)

        frame_id += 1
        list_tfl.append((cur_container.traffic_light_red, cur_container.traffic_light_green))


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
