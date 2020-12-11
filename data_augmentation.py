import numpy as np
import os
import scipy.io.wavfile as wav
from util import save_wave_file
# import matplotlib.pyplot as plt


# Add normal noise according to amplituede
def _add_noise(audio, noise_factor=0):

    if noise_factor == 0:
        noise_factor = (np.max(audio) - np.min(audio)) / 20
    
    return audio + noise_factor * np.random.randn(audio.shape[0])



# Linear timeshifting 
def _time_shift(audio, shift_type):

    length = audio.shape[0]
    xp = np.array([i for i in range(length)])


    if shift_type == 'up_first':
        midpoint = length / 2 * 1.1
    if shift_type == 'down_first':
        midpoint = length / 2 * 0.9
    
    half_length = int(length / 2)
    x = np.hstack((
        np.linspace(0, midpoint, half_length + 1)[0:-1],
        np.linspace(midpoint, length, length-half_length)))


    return np.interp(x, xp, audio)

        


# Change pitc
def _pitch_shift(audio, shift_type, n_step = 30):
    
    fft = np.fft.fft(audio)
    length = fft.shape[0]

    if shift_type == 'pitch_up':
        fft_ = np.hstack(([0] * n_step ,fft[0:length-n_step]))
    if shift_type == 'pitch_down':
        fft_ = np.hstack((fft[n_step:length], [0]*n_step))

    return np.fft.ifft(fft_)
    



def dataset_augmentation(path):
    class_str_list = os.listdir(path)
    for dir_name in class_str_list:
        cur_dir = os.path.join(path, dir_name + '/')
        for wav_file_name in os.listdir(cur_dir):
            if 'aug' not in wav_file_name:
                
                wav_file_path = cur_dir + '/' + wav_file_name
                (_, sig) = wav.read(wav_file_path)

                # Add noise and save
                add_noise_file_name = 'aug_add_noise_' + wav_file_name
                if add_noise_file_name not in os.listdir(cur_dir):
                    add_noise_sig = _add_noise(sig)
                    add_noise_file_path = cur_dir + '/' + add_noise_file_name
                    save_wave_file(add_noise_sig.astype(np.int32), add_noise_file_path)

                # time shift and save
                for shift_type in ['up_first', 'down_first']:
                    time_shift_file_name = 'aug_' + shift_type + '_' + wav_file_name
                    if time_shift_file_name not in os.listdir(cur_dir):
                        time_shift_sig = _time_shift(sig, shift_type)
                        time_shift_file_path = cur_dir + '/' + time_shift_file_name
                        save_wave_file(time_shift_sig.astype(np.int32), time_shift_file_path)

                # Pitch shift and save
                for shift_type in ['pitch_up', 'pitch_down']:
                    pitch_shift_file_name = 'aug_' + shift_type + '_' + wav_file_name
                    if pitch_shift_file_name not in os.listdir(cur_dir):
                        pitch_shift_sig = _pitch_shift(sig, shift_type)
                        pitch_shift_file_path = cur_dir + '/' + pitch_shift_file_name
                        save_wave_file(pitch_shift_sig.astype(np.int32), pitch_shift_file_path)

                

def remove_augmentation_samples(path):
   
    class_str_list = os.listdir(path)
   
    for dir_name in class_str_list:
        cur_dir = os.path.join(path, dir_name + '/')
        for wav_file_name in os.listdir(cur_dir):
            if 'aug' in wav_file_name:
                
                wav_file_path = cur_dir + '/' + wav_file_name 
                os.remove(wav_file_path)



if __name__ == '__main__':
    
    import pyaudio
    import struct
    from config import _AUDIO_FRAME_RATE, _AUDIO_DATA_WIDTH, _AUDIO_CHANNELS
    # from util import audio_interp

    """
    p = pyaudio.PyAudio()

    PA_FORMAT = p.get_format_from_width(_AUDIO_DATA_WIDTH)
    stream = p.open(
        format = PA_FORMAT,
        channels = _AUDIO_CHANNELS,
        rate = _AUDIO_FRAME_RATE,
        input = False,
        output = True)
    test_wav_path = 'audio_data/class_2/2_0.wav'

    # Read wavefile
    (rate, sig) = wav.read(test_wav_path)

    # sig_ = _add_noise(sig)
    # sig_ = _time_shift(sig, 'down_first')
    # sig_ = audio_interp(sig)
    sig_ = _pitch_shift(sig, 'down')
    sig_ = sig_.astype(np.int32).tolist()
    sig = sig.tolist()
    output_bytes = struct.pack('h'*len(sig_), *sig_)
    
    stream.write(output_bytes)

    stream.stop_stream()
    stream.close()
    """
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", action="store_true")
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()


    data_dir = 'audio_data'

    # dataset_augmentation(data_dir)
    if args.remove:
        remove_augmentation_samples(data_dir)

    if args.create:
        dataset_augmentation(data_dir)
