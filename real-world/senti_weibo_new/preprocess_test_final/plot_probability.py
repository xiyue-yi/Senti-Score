import os
import json
import sys
import random
import pandas as pd
import numpy as np
import pdb
import re

import os
import time
import argparse
import matplotlib.pyplot as plt

topic_name = "权力的游戏"
probability_dir = os.path.join("./probability_data",topic_name)
model_index = 12
date = "2019-04-01"
probability_file = probability_dir+'/'+str(model_index)+'/'+date+'.txt'
probability_data = np.loadtxt(probability_file)
x = probability_data[:,0]
y = probability_data[:,1]

plt.scatter(x,y)
plt.show()            

def main():
    pass
    


if __name__ == "__main__":
    main()