from rousseau_grid import PianoGrid
from grid_utils import get_sample_frame
import matplotlib.pyplot as plt

sample_img = get_sample_frame()

grid = PianoGrid()
# grid.detect_grid(sample_img)
grid.draw_grid(sample_img, use_global_polygons=True)

plt.imshow(sample_img)
plt.show()