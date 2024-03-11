import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

x = np.random.uniform(0, 100, 2000)
y = np.random.uniform(0, 50, 2000)
ID = np.random.randint(0,100,2000)

fig, ax = plt.subplots(figsize=(10, 8),dpi = 80)
scatter = ax.scatter(x,
                     y,
                    c = ID)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(*scatter.legend_elements(),
          loc="center left",
          title='ID',
          bbox_to_anchor=(1, 0.5)
         )
ax.ticklabel_format(useOffset=False)
ax.tick_params(axis = 'x',labelrotation = 45)
plt.show()
