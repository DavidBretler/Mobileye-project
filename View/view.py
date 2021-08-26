import numpy as np
from matplotlib import pyplot as plt
import matplotlib._png as png
PATH2 = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2'
import Model.SFM as SFM


def visualize(cur_frame,  red_lights, green_lights, lst_dist,frame_id):
    fig,image = plt.subplots()
    image.set_title('Frame ID:'+str(frame_id))
    image.imshow(png.read_png_int(PATH2+'\\'+cur_frame))
    red_x_lst = [x[0] for x in red_lights]
    red_y_lst = [x[1] for x in red_lights]

    green_x_lst = [x[0] for x in green_lights]
    green_y_lst = [x[1] for x in green_lights]

    image.plot(red_x_lst, red_y_lst, 'r+')
    image.plot(green_x_lst, green_y_lst, 'g+')
    for i in range(len(red_x_lst)):
         image.text(red_x_lst[i], red_y_lst[i],lst_dist[i], color='r')

    for i in range(len(green_x_lst)):
        image.text(green_x_lst[i], green_y_lst[i], lst_dist[i+len(red_x_lst)-1], color='r')
    plt.show()

