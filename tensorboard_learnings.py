# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 09:55:23 2020

@author: engro
"""


from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("TensorFlow version: ", tf.__version__)
# assert version.parse(tf.__version__).release[0] >= 2, \
#     "This notebook requires TensorFlow 2.0 or above."