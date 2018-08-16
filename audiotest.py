import os
import numpy as np
from pydub import AudioSegment

AUDIODIR = 'datasets/audio/'
def sec2human(sec):
	h = sec//3600
	m = (sec-h*3600)//60
	s = (sec-h*3600-m*60)
	return '{}:{}:{}'.format(h,m,s)

song = AudioSegment.from_mp3(os.path.join(AUDIODIR,'Temperature_SeanPaul.mp3'))

# FrameRate reduced, so audio quality is still acceptable
FrameRate = 8000
song = song.set_frame_rate(FrameRate)

# Each sample is [ch1(sample_width), ch2(sample_width)]
raw_data = song.raw_data
encoded = np.fromstring(song.raw_data, dtype=np.uint16)
decoded = encoded.tostring()

# Check that new song can be created identically to old song, and that data is converted correctly to and from bytestream
newsong = AudioSegment(data = decoded,
                       sample_width = song.sample_width,
                       frame_rate = song.frame_rate,
                       channels = song.channels)
assert(song==newsong)

# Splits song into seperate channels, just for fun
ChannelDict = {'CH{}'.format(x+1): encoded[x::song.sample_width] \
               for x in range(song.channels)}

for key, val in ChannelDict.items():
    name = os.path.join(AUDIODIR,'Temperature_{}.mp3'.format(key))
    song = AudioSegment(data = val.tostring(),
                       sample_width = song.sample_width,
                       frame_rate = song.frame_rate,
                       channels = 1)

    with open(name, 'wb') as out_f:
        song.export(out_f, format='mp3')