import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

frames = np.load(sys.argv[1])['obs']

fig, ax = plt.subplots()
ax.axis('off')

im = ax.imshow(frames[0])

def update(frame):
    im.set_data(frames[frame])
    return [im]

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=33, blit=True)
    
plt.show()