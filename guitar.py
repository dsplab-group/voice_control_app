import numpy as np

from config import _GUITAR_RATE, _AUDIO_DATA_MAX

K = 0.8
buffer_length = [800,700,650,600,550,500,450,400]


def generate_guitar_music(note_list, gap_time, duration_time):
    
    gap_n = int(gap_time * _GUITAR_RATE)
    note_length = int(duration_time * _GUITAR_RATE)
    output = np.zeros(note_length)
    output_start_id = gap_n
    for note in note_list:

        N = buffer_length[int(note) - 1]
        x = 20000.0 * np.hstack((np.random.rand(N), np.zeros(note_length-N)))
        x = x.astype(int)

        y = np.zeros(note_length)

        for i in range(note_length):
            f_val_1 = y[i-N] if i >= N else 0.0
            f_val_2 = y[i-1] if i > 0 else 0.0
            y[i] = x[i] + K *(0.5*f_val_1 + 0.5*f_val_2)
        
        output = np.hstack((output[:output_start_id],
                            output[output_start_id:]+y[:note_length-gap_n], y[note_length-gap_n:]))

        output_start_id = output_start_id + gap_n
    output= np.clip(output.astype(int), -_AUDIO_DATA_MAX, _AUDIO_DATA_MAX)


    return output



if __name__ == '__main__':

    import pyaudio
    import struct
    from config import _AUDIO_DATA_WIDTH, _AUDIO_CHANNELS
    
    p = pyaudio.PyAudio()
    PA_FORMAT = p.get_format_from_width(_AUDIO_DATA_WIDTH)

    stream = p.open(
        format=pyaudio.paInt16,
        channels=_AUDIO_CHANNELS,
        rate=_GUITAR_RATE,
        input=False,
        output=True) 
    

    play_notes = [1, 2, 3, 4]

    music = generate_guitar_music(play_notes, 0.5, 1.0)
    print(music.shape)

    output_bytes = struct.pack('h' * music.shape[0], *music)

    stream.write(output_bytes)

    stream.stop_stream()
    stream.close()