import numpy as np
import matplotlib.pyplot as plt

data = np.load('reward_action_per_ep_60.npy')
plt.plot(data)
plt.show()
#print(np.max(data))
#print(data)