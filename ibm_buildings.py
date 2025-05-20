import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 100, 100)
y = np.linspace(0, 200, 200)
z = np.linspace(0, 300, 300)

buildings = [
    [40, 50, 40,  60, 20],
    [60, 80, 90, 100, 10],
]

num_cells = x.shape[0] * y.shape[0]
num_levels = z.shape[0]

X, Y, Z = np.meshgrid(x, y, z)

cell_x = X[:,:,0].flatten()
cell_y = Y[:,:,0].flatten()
lev_z  = Z.reshape(-1, Z.shape[-1])
mask = np.zeros((num_cells, num_levels), dtype=bool)

for k in range(mask.shape[1]):
    for building in buildings:
        xmin, xmax, ymin, ymax, top = building
        mask[:, k] = np.where( (cell_x >= xmin) & (cell_x <= xmax) & (cell_y >= ymin) & (cell_y <= ymax) & (lev_z[:,k] <= top), True, mask[:,k])

M = mask.reshape((x.shape[0], y.shape[0], z.shape[0]))

k = 20
plt.figure(1); plt.clf(); plt.show(block=False)
plt.scatter(cell_x, cell_y, s=1)
plt.scatter(cell_x[mask[:,k]], cell_y[mask[:,k]], s=2)
plt.draw()
