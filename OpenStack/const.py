DATA_DIR = "../../data/openstack/"
OUTPUT_DIR = "../output/openstack/"
PARSER = 'spell'


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(0,1, size=100)
y = np.random.normal(3,2, size=100)

sns.kdeplot(x)
#sns.distplot(y)
plt.show()