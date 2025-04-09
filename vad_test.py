import torchaudio
from speechbrain.pretrained import VAD

vad = VAD.from_hparams(source= "speechbrain/vad-crdnn-libriparty")

signal, fs = torchaudio.load("example_vad_music.wav")
print(" ------------------------------------------------------------------   ")
if fs!=16000:
    resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)
    signal = resampler(signal)

boundaries = vad.get_speech_segments('example_vad_music.wav')
vad.save_boundaries(boundaries)
i = 0
for segment in boundaries:
	print(i)
	i=i+1
