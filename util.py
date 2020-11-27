import numpy as np
import wave
import matplotlib.pyplot as plt
import io
import pylab
import cv2 as cv
from scipy.signal import butter, lfilter
import struct
import os

from config import _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS, _AUDIIO_CUT_THRESHOLD, _AUDIO_CHANNELS, _AUDIO_FRAME_RATE, _AUDIO_DATA_WIDTH, _EXTEND_NUM, _FILTER_ORDER, _LOW_PASS_CUTOFF, _SVM_IMAGE_HEIGHT, _SVM_IMAGE_WIDTH

def get_wav_info(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate



def get_arr_from_audio(audio_data, f=8000, showImg=False, Transfer=False, shape_type='SVM'):
    plt.specgram(audio_data, Fs=f)
    plt.axis('off')
    fig = plt.gcf()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inchers='tight')
    buf.seek(0)

    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if shape_type == 'CNN':
        img = cv.resize(img, (_IMAGE_HEIGHT, _IMAGE_WIDTH))
    elif shape_type == 'SVM':
        img = cv.resize(img, (_SVM_IMAGE_HEIGHT, _SVM_IMAGE_WIDTH))

    if showImg:
        cv.imshow('1', img)
        cv.waitKey(0)

    if Transfer:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if shape_type == 'CNN':
        arr = img.reshape(-1, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNELS)
    if shape_type == 'SVM':
        arr = img.reshape(-1)


    
    plt.clf()
    plt.close()

    
    arr = arr.astype(np.float32) / 255.0
    

    return arr

def save_img_from_audio(audio_data, savepath, f=8000, showImg=False):
    plt.specgram(audio_data, Fs=f)
    plt.axis('off')

    plt.savefig(savepath, bbox_inches='tight')
    plt.clf()
    plt.close()
    

def cut_audio(audio_data):
    startID = 0
    endID = audio_data.shape[0]


    for data in audio_data:

        if data > _AUDIIO_CUT_THRESHOLD:
            break

        startID = startID + 1

    startID = max(0, startID - _EXTEND_NUM)
    
    for data in reversed(audio_data):

        if data > _AUDIIO_CUT_THRESHOLD:
           break

        endID = endID - 1

    endID = min(audio_data.shape[0], endID + _EXTEND_NUM)

    return audio_data[startID:endID]


def save_wave_file(audio_data, path):
    wf = wave.open(path, 'w')
    wf.setnchannels(_AUDIO_CHANNELS)
    wf.setsampwidth(_AUDIO_DATA_WIDTH)
    wf.setframerate(_AUDIO_FRAME_RATE)

    

    byte_string = struct.pack('h'*audio_data.shape[0], *audio_data)

    wf.writeframes(byte_string)

    wf.close()


def butter_lowpass(cutoff=_LOW_PASS_CUTOFF, fs=_AUDIO_FRAME_RATE, order=_FILTER_ORDER):

    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return b, a

def data_matrix_from_path(path='sample_data/', Transfer=False):
    
    class_str_list = os.listdir(path)
    X_list = []
    y = []

    for id_, dir_name in enumerate(class_str_list):
        cur_dir = os.path.join(path, dir_name + '/')
        for png_file_name in os.listdir(cur_dir):
            img = cv.imread(cur_dir + png_file_name)
            img = cv.resize(img, (_SVM_IMAGE_HEIGHT, _SVM_IMAGE_WIDTH))

            if Transfer:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


            img = img.astype(np.float32) / 255.0
            X_list.append(img.reshape(-1))
            y.append(id_)

    X = np.vstack(X_list)
    y = np.hstack(y)

    return X, y
            
