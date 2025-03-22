import numpy as np
import matplotlib.pyplot as plt
import torch
import random
# from pycg import color

def vis_pcs(pcl_lst, S=3, vis_order=[2,0,1], bound=1):
    fig = plt.figure(figsize=(3 * len(pcl_lst), 3))
    num_col = len(pcl_lst)
    for idx, pts in enumerate(pcl_lst):
        ax1 = fig.add_subplot(1, num_col, 1 + idx, projection='3d')
        rgb = None
        psize = S 
        
        # normalize the points
        if pts.size > 0:
            if np.abs(pts).max() > bound:
                pts = pts / np.abs(pts).max()
        
        ax1.scatter(pts[:, vis_order[0]], -pts[:, vis_order[1]], pts[:, vis_order[2]], s=psize, c=rgb)
        ax1.set_xlim(-bound, bound)
        ax1.set_ylim(-bound, bound)
        ax1.set_zlim(-bound, bound)
        ax1.grid(False)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig) # close the figure to avoid memory leak
    return image_from_plot


def random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
