import pyaudio
import os
import time
import cv2 as cv
from config import _AUDIO_CHANNELS, _AUDIO_DATA_WIDTH, _AUDIO_VALID_THRESHOLD, _AUDIO_FRAME_RATE, _BLOCKLEN, _USE_FILTER, _AUDIO_MAX_GAP
from util import cut_audio, save_img_from_audio, butter_lowpass, save_wave_file
from scipy.signal import lfilter
import numpy as np

from config import target_dict



print('Please enter the png file dictionary (Default: sample_data/): ')
PNG_OUTPUT_DIR = input()
if PNG_OUTPUT_DIR == '':
    PNG_OUTPUT_DIR = 'sample_data/'
print('Please enter the wave file dictionary (Default: audio_data/')
WAV_OUTPUT_DIR = input()
if WAV_OUTPUT_DIR == '':
    WAV_OUTPUT_DIR = 'audio_data/'

print('Please enter your personal label: ')
pLabel = input()
if pLabel != '':
    pLabel = pLabel + '_'

if not os.path.exists(PNG_OUTPUT_DIR):
    os.mkdir(PNG_OUTPUT_DIR)

if not os.path.exists(WAV_OUTPUT_DIR):
    os.mkdir(WAV_OUTPUT_DIR)

for class_name in target_dict.values():
    if not os.path.exists(os.path.join(PNG_OUTPUT_DIR, f'class_{class_name}')):
        os.mkdir(os.path.join(PNG_OUTPUT_DIR, f'class_{class_name}'))
    if not os.path.exists(os.path.join(WAV_OUTPUT_DIR, f'class_{class_name}')):
        os.mkdir(os.path.join(WAV_OUTPUT_DIR, f'class_{class_name}'))

recording = False
detected = False
gap_time = 0
block_buffer= []

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
    
print('You can enter following char: ')
for (k, v) in target_dict.items():
    print(k, '=======>', v)

print('Press q to quit')

if _USE_FILTER:
    b, a = butter_lowpass()

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

while True:

    print('Please enter: ')
    char = input()

    if char =='q':
        break

    if char not in target_dict.keys():
        print('Wrong Input')
        continue

    print('You are recording ', '\'',target_dict[char], '\'')
    recording = True
    stream.start_stream()
    while True:
        time.sleep(0.1)

        if not recording:
            stream.stop_stream()
            print('Complete: ', char)

            audio_sequence = np.hstack(block_buffer)            
            audio_sequence = cut_audio(audio_sequence)

            if _USE_FILTER:
                audio_sequence = lfilter(b, a, audio_sequence)

            block_buffer = []
            output_class_dir = os.path.join(PNG_OUTPUT_DIR, f'class_{target_dict[char]}')
            
            fId = 0
            while f'{pLabel}{target_dict[char]}_{fId}.png' in os.listdir(output_class_dir):
                fId = fId + 1

            save_path = output_class_dir + '/' + f'{pLabel}{target_dict[char]}_{fId}.png'

            save_img_from_audio(audio_sequence, save_path)

            output_class_dir = os.path.join(WAV_OUTPUT_DIR, f'class_{target_dict[char]}')
            fId = 0
            while f'{pLabel}{target_dict[char]}_{fId}.wav' in os.listdir(output_class_dir):
                fId = fId + 1
            save_path = output_class_dir + '/' + f'{pLabel}{target_dict[char]}_{fId}.wav'
            save_wave_file(audio_sequence, save_path)
            
            print('Saved')

            break

print('Thank you for recording!')