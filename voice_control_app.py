import pyaudio
import numpy as np
import struct
import time
import joblib
import pyaudio, struct

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

    import os

from config import target_dict
from util import cut_audio, audio_interp
from config import target_dict, _BLOCKLEN, _MUSIC_BLOCK_LEN, _AUDIO_CHANNELS, _AUDIO_FRAME_RATE, _AUDIO_VALID_THRESHOLD, _AUDIO_DATA_WIDTH, _AUDIO_MAX_GAP, _GUITAR_RATE
from mfcc import mfcc_feature_pyramid
from piano import generate_piano_music
from guitar import generate_guitar_music

svm_model_path = 'models/svm_mfcc_data_aug_model'
my_model = joblib.load(svm_model_path)

# stream init
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(_AUDIO_DATA_WIDTH)
# voice recording flag 
voice_detected = False
voice_recording = False
gap_time = 0

# recording data buffer
voice_block_buffer = []

def callback(in_data, frame_count, time_info, flag):
    global voice_block_buffer, voice_detected, voice_recording, gap_time
    signal_block = np.frombuffer(in_data, dtype=np.int16)
    # Record if the value is greater the what we defined
    audio_valid = (np.max(signal_block)-np.min(signal_block)) > _AUDIO_VALID_THRESHOLD

    
    
    if not voice_detected and audio_valid:
        voice_detected = True
    if voice_detected and gap_time < _AUDIO_MAX_GAP and not audio_valid:
        gap_time = gap_time + 1
    if voice_detected and gap_time == _AUDIO_MAX_GAP and not audio_valid:
        voice_detected = False
        voice_recording = False
        gap_time = 0
    if voice_detected:
        voice_block_buffer.append(signal_block)

    return(signal_block, pyaudio.paContinue)
 


voice_detect_stream = p.open(
    format=PA_FORMAT,
    channels=_AUDIO_CHANNELS,
    rate=_AUDIO_FRAME_RATE,
    input=True,
    output=False,
    stream_callback=callback,
    frames_per_buffer=_BLOCKLEN)



piano_play_stream = p.open(
    format=pyaudio.paInt16,
    channels=_AUDIO_CHANNELS,
    rate=_AUDIO_FRAME_RATE,
    input=False,
    output=True)
    #frames_per_buffer=_MUSIC_BLOCK_LEN)

guitar_play_stream = p.open(
    format=pyaudio.paInt16,
    channels=_AUDIO_CHANNELS,
    rate=_GUITAR_RATE,
    input=False,
    output=True
)

# instrument initialize
current_instrument = 'piano'
instruments_list = ['guitar', 'piano']

while True:

    # Strat record voice
    voice_detect_stream.start_stream()
    record_words_list = []
    voice_recording = True
    voice_block_buffer = []

    if_start = False

    print(f'default instrument: {current_instrument}')
    print('strat')

    while True:

        time.sleep(0.1)

        if not voice_recording:
            voice_detect_stream.stop_stream()

            # get saved audio data
            voice_sequence = np.hstack(voice_block_buffer)
            voice_sequence = cut_audio(voice_sequence)

            # calculate features
            voice_arr = audio_interp(voice_sequence)
            voice_mfcc_feat = mfcc_feature_pyramid(voice_arr)

            # Predict what users said
            res = my_model.predict(voice_mfcc_feat.reshape(1,-1))[0]
            record_word = list(target_dict.values())[res]

            
            print(f'voice detected: {record_word}')

            # start record all the words
            if not if_start and record_word == 'begin':
                if_start = True
            elif record_word == 'end':
                if_start = False
                break
            elif if_start:
                record_words_list.append(record_word)

            print('words list:')
            print(record_words_list)
            voice_block_buffer = []
            voice_detect_stream.start_stream()
            voice_recording = True

    # Start playing music
    note_words = ['1', '2', '3', '4', '5', '6', '7', '8']
    print('play')
    #music_play_stream.start_stream()
    record_notes = []
    for record_word in record_words_list:
        if record_word in instruments_list:
            if record_word != current_instrument:

                # play music
                print(f'play as {current_instrument}')

                if current_instrument == 'piano':
                    music = generate_piano_music(record_notes, 1.0, 1.0)
                    output_bytes = struct.pack('h'*music.shape[0], *music)
                    piano_play_stream.start_stream()
                    piano_play_stream.write(output_bytes)
                    piano_play_stream.stop_stream()
                if current_instrument == 'guitar':
                    music = generate_guitar_music(record_notes, 1.0, 1.0)
                    output_bytes = struct.pack('h'*music.shape[0], *music)
                    guitar_play_stream.start_stream()
                    guitar_play_stream.write(output_bytes)
                    guitar_play_stream.write(output_bytes)
                
                # change instrument
                current_instrument = record_word
                print(f'instrement change to {current_instrument}')

                record_notes = []

            continue
        if record_word in note_words:
            record_notes.append(record_word)

    print(record_notes)
    if not len(record_notes) == 0:
        if current_instrument == 'piano':
            music = generate_piano_music(record_notes, 1.0, 1.0)
            output_bytes = struct.pack('h'*music.shape[0], *music)
            piano_play_stream.start_stream()
            piano_play_stream.write(output_bytes)
            piano_play_stream.stop_stream()
        if current_instrument == 'guitar':
            music = generate_guitar_music(record_notes, 1.0, 1.0)
            output_bytes = struct.pack('h'*music.shape[0], *music)
            guitar_play_stream.start_stream()
            guitar_play_stream.write(output_bytes)
            guitar_play_stream.stop_stream()
                 
        record_notes = []
    


