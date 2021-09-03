import numpy as np
import matplotlib.pyplot as plt

data = np.load('col_num_per_ep_66.npy')
plt.plot(data)
plt.show()
#print(np.max(data))
#print(data)