import numpy as np
from python_speech_features import mfcc

from config import _AUDIO_LENGTH, _AUDIO_FRAME_RATE

mfcc_frame_len_bank =[
    256,
    512,
    1024,
]

def mfcc_feature_pyramid(audio):
    if audio.shape[0] != _AUDIO_LENGTH:
        return

    feat_pyramid = []
    for winlen in mfcc_frame_len_bank:
        winlen_ = winlen / _AUDIO_FRAME_RATE
        feat_pyramid.append(np.array(mfcc(audio, _AUDIO_FRAME_RATE, winlen=winlen_, winstep=winlen_/2,nfft=1024)).reshape(-1))

    return np.hstack(feat_pyramid).reshape(-1)

    

