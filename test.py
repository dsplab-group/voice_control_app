import pyaudio
# import tensorflow as tf
import numpy as np
import struct
import time
import joblib
import cv2 as cv

# from cnn_model import get_model
from util import cut_audio, get_arr_from_audio, save_wave_file, audio_interp
from mfcc import mfcc_feature_pyramid

from config import _AUDIO_CHANNELS, _AUDIO_DATA_WIDTH, _AUDIO_VALID_THRESHOLD, _AUDIO_FRAME_RATE, _BLOCKLEN, _SVM_IMAGE_HEIGHT, _SVM_IMAGE_WIDTH, _AUDIO_MAX_GAP

import os

from config import target_dict

target_val = list(target_dict.values())

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SAVE_AUDIO = False
SAVE_IMAGE = True
audio_saved_count = 0
image_saved_count = 0

MODEL_TPYE = 'SVM'

svm_model_path = 'models/svm_mfcc_data_aug_model'
test_data = 'test_data/'


if MODEL_TPYE == 'SVM':
    my_model = joblib.load(svm_model_path)
block_buffer = []


recording = True
detected = False
gap_time = 0

def callback(in_data, frame_count, time_info, flag):
    global block_buffer, detected, recording, gap_time
    signal_block = np.frombuffer(in_data, dtype=np.int16)
    # Record if the value is greater the what we defined
    audio_valid = (np.max(signal_block)-np.min(signal_block)) > _AUDIO_VALID_THRESHOLD

    
    
    if not detected and audio_valid:
        detected = True
    if detected and gap_time < _AUDIO_MAX_GAP and not audio_valid:
        gap_time = gap_time + 1
    if detected and gap_time == _AUDIO_MAX_GAP and not audio_valid:
        detected = False
        recording = False
        gap_time = 0
    if detected:
        block_buffer.append(signal_block)

    return(signal_block, pyaudio.paContinue)
 

p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(_AUDIO_DATA_WIDTH)
stream = p.open(
    format = PA_FORMAT,
    channels = _AUDIO_CHANNELS,
    rate = _AUDIO_FRAME_RATE,
    input = True,
    output = False,
    stream_callback=callback,
    frames_per_buffer=_BLOCKLEN)

print('**start**')

stream.start_stream()

while True:
    time.sleep(0.1)

    if not recording:
        stream.stop_stream()

        print('Voice detected')
        
        audio_sequence = np.hstack(block_buffer)
        audio_sequence = cut_audio(audio_sequence)
    
        # img_arr = get_arr_from_audio(audio_sequence,showImg=True, Transfer=False)

        audio_arr = audio_interp(audio_sequence)
        audio_mfcc_feat = mfcc_feature_pyramid(audio_arr)

        """
        if MODEL_TPYE == 'CNN':
            res_arr = my_model.predict(img_arr)
            res = np.where(res_arr == np.max(res_arr))[1][0]
        """

        res = 0

        if MODEL_TPYE == 'SVM':
            res = my_model.predict(audio_mfcc_feat.reshape(1,-1))[0]
        print(target_val[res])        

        if SAVE_AUDIO:
            wave_file_name = f'testdata/audio_save_{audio_saved_count}_{res}.wav'
            save_wave_file(test_data + audio_sequence, wave_file_name)
            audio_saved_count = audio_saved_count + 1
        
        """
        if SAVE_IMAGE:
            image_file_name = f'testdata/image_save_{image_saved_count}_{res}.png'
            img_arr = img_arr.reshape(_SVM_IMAGE_HEIGHT, _SVM_IMAGE_WIDTH, 3) * 255
            cv.imwrite(test_data + image_file_name, img_arr.astype(np.int16))
            image_saved_count = image_saved_count + 1
        """

        block_buffer = []
        recording = True
        stream.start_stream()

        
    

stream.stop_stream()
stream.close()
p.terminate()