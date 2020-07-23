DATA_DIR = "../../data/openstack/"
OUTPUT_DIR = "../output/openstack/"
PARSER = 'spell'


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
x = np.random.normal(0,1, size=100)
y = [10] * 5 + [1] * 50 + [5] * 10 + [2] * 20
sns.kdeplot(y)
sns.distplot(y, hist=False, kde=True)
plt.show()