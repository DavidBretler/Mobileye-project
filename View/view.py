import numpy as np
from matplotlib import pyplot as plt
import matplotlib._png as png

from Model import SFM

PATH2 = r'C:\Users\ddkil\OneDrive\מסמכים\GitHub\mobileye-part-b-davidteam2'



def visualize(cur_container,frame_id):
    fig,image = plt.subplots()
    image.set_title('Frame ID:'+str(frame_id))
    image.imshow(cur_container.img)

    red_x_lst = [x[0] for x in cur_container.traffic_light_red]
    red_y_lst = [x[1] for x in cur_container.traffic_light_red]

    green_x_lst = [x[0] for x in cur_container.traffic_light_green]
    green_y_lst = [x[1] for x in cur_container.traffic_light_green]

    image.plot(red_x_lst, red_y_lst, 'r+')
    image.plot(green_x_lst, green_y_lst, 'g+')

    for i in range(len(red_x_lst)):
        if cur_container.valid[i]:
           image.text(red_x_lst[i], red_y_lst[i], r'{0:.1f}'.format(cur_container.traffic_lights_3d_location[i,2]), color='b')

    for i in range(len(green_x_lst)):
        if cur_container.valid[i+len(red_x_lst)]:
           image.text(green_x_lst[i], green_y_lst[i], r'{0:.1f}'.format( cur_container.traffic_lights_3d_location[(i+len(red_x_lst)), 2]), color='b')
    plt.show()

