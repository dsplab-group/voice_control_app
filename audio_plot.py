import argparse

import wave
import struct
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import get_model
from util import get_arr_from_audio
import tensorflow as tf


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

weight_save_path = 'models/model2_checkpoint'

# my_model = get_model()
# my_model.load_weights(weight_save_path)


parser = argparse.ArgumentParser(description='Audio file path')

parser.add_argument('--filepath', '-f', required=True)
args = parser.parse_args()

wf = None

try:
    wf = wave.open(args.filepath)
except IOError:
    print('No such file named: {args.filepath}')

if wf:
    input_byte = wf.readframes(-1)

    input_tuple = struct.unpack('h'*wf.getnframes(), input_byte)

    img_arr = get_arr_from_audio(input_tuple)
    # res = my_model.predict(img_arr)

    plt.plot(input_tuple)

    plt.show()

wf.close()