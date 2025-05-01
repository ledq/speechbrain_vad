import onnxruntime
import numpy as np
import pyaudio
import queue

session = onnxruntime.InferenceSession("silero_vad.onnx")

# Detect input names
input_names = [i.name for i in session.get_inputs()]
print(f"Detected input names: {input_names}")

input_name_audio = input_names[0]  # 'input'
input_name_state = input_names[1]  # 'state'
input_name_sr = input_names[2]     # 'sr'

# Detect type for state
state_dtype = np.int64 if session.get_inputs()[1].type == 'tensor(int64)' else np.float32
sr_dtype = np.int64 if session.get_inputs()[2].type == 'tensor(int64)' else np.float32

# Initialize state and sampling rate
state = np.zeros((2, 1, 128), dtype=state_dtype)  # packed hidden and cell
sr = np.array([16000], dtype=sr_dtype)  # sampling rate

# Microphpone
sample_rate = 16000  
frames_per_buffer = 512  # or 256 for smaller chunks
q = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    q.put(in_data)
    return (None, pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                input=True,
                input_device_index=0,  # <== PUT CORRECT INDEX
                frames_per_buffer=frames_per_buffer,
                stream_callback=callback)
stream.start_stream()

print("Listening...")


try:
    while True:
        if not q.empty():
            raw_audio = q.get()
            samples = np.frombuffer(raw_audio, dtype=np.int16)
            #print(f"First samples: {samples[:10]}")
            # Convert raw audio to float32 in [-1.0, 1.0]
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            #print(f"Mean amplitude after scaling: {np.mean(np.abs(audio_np))}")
            audio_np = np.expand_dims(audio_np, axis=0)  # (1, N)

            # Run inference
            outputs = session.run(None, {
                input_name_audio: audio_np,
                input_name_state: state,
                input_name_sr: sr
            })

            # Read outputs
            speech_prob = outputs[0][0][0]  # Speech probability
            state = outputs[1]              # Updated packed hidden+cell state

            # Print speech probability
            print(f"Speech probability: {speech_prob:.3f}")

except KeyboardInterrupt:
    print("\nStopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
