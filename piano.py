import numpy as np
from scipy import signal
from math import sin, cos, pi

from config import _MUSIC_BLOCK_LEN, _AUDIO_FRAME_RATE, _AUDIO_DATA_MAX

f0 = 440

freq_list = f0*np.array([pow(2, -4/12), pow(2, -3/12), pow(2, -2/12) ,pow(2,-1/12), 1, pow(2,1/12), pow(2,2/12), pow(2,3/12)])

def generate_piano_music(words_sequence, gap_time, duration_time):

    gap_n = int(gap_time * _AUDIO_FRAME_RATE)
    # data length per
    note_length = int(duration_time * _AUDIO_FRAME_RATE)
    

    x = np.zeros(note_length)
    x[0] = 10000.0

    output = np.zeros(note_length)
    output_start_id = gap_n
    r = 0.01**(1.0/(duration_time*_AUDIO_FRAME_RATE))

    for words in words_sequence:
        f1 = freq_list[int(words)-1]


        om1 = 2.0 * pi * float(f1)/_AUDIO_FRAME_RATE
        a = [1, -2*r*cos(om1), r**2]
        b = [r*sin(om1)]

        sig = np.zeros(2)

        y, _ = signal.lfilter(b, a, x, zi=sig)

        output = np.hstack((output[:output_start_id],
                            output[output_start_id:]+y[:note_length-gap_n], y[note_length-gap_n:]))

        output_start_id = output_start_id + gap_n
    output= np.clip(output.astype(int), -_AUDIO_DATA_MAX, _AUDIO_DATA_MAX)
    return output

if __name__ == '__main__':

    import pyaudio
    import struct
    from config import _AUDIO_DATA_WIDTH, _AUDIO_FRAME_RATE, _MUSIC_BLOCK_LEN, _AUDIO_CHANNELS
    import matplotlib.pyplot as plt
    p = pyaudio.PyAudio()
    PA_FORMAT = p.get_format_from_width(_AUDIO_DATA_WIDTH)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=_AUDIO_CHANNELS,
        rate=_AUDIO_FRAME_RATE,
        input=False,
        output=True) 
    
    note_list = [1,2,3,4]


    duration_time = 1.0
    gap_time = 0.5
    

    out_array = generate_piano_music(note_list, gap_time, duration_time).astype(np.int16)
    

    out_bytes = struct.pack('h'*out_array.shape[0], *out_array)
    stream.write(out_bytes)
    # print(out_array.shape)
    x = np.array([i for i in range(out_array.shape[0])])
    # plt.plot(x, out_array)
    # plt.show()

    stream.stop_stream()
    stream.close()