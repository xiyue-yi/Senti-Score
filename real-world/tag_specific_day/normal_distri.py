import os
import json
import sys
import random
import numpy as np
import math
from math import sqrt as sqrt

x=12
u=14
sig2=4.2
sig = sqrt(sig2)
a = np.exp(-(x - u) ** 2 / (2 * sig2))
b = (math.sqrt(2 * math.pi) * sig)
y_sig = np.exp(-(x - u) ** 2 / (2 * sig2)) / (math.sqrt(2 * math.pi) * sig)
print(a,b,y_sig)


x=0.6
u=0.7
sig2=0.0105
sig = sqrt(sig2)
a = np.exp(-(x - u) ** 2 / (2 * sig2))
b = (math.sqrt(2 * math.pi) * sig)
y_sig = np.exp(-(x - u) ** 2 / (2 * sig2)) / (math.sqrt(2 * math.pi) * sig)
print(a,b,y_sig)

