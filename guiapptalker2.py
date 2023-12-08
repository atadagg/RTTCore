import tkinter as tk
from threading import Thread
import os
from google.cloud import speech
import pyaudio
from six.moves import queue
from google.cloud import translate_v2 as translate
from google.cloud import texttospeech
import io
import pygame
import cv2
import multiprocessing


# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

translate_client = translate.Client()
camera_running = False



class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't overflow
            # while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)



def show_camera():
    global camera_running
    camera_running = True

    cap = cv2.VideoCapture(0)

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow('Video Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    camera_running = False

def run_camera_process():
    p = multiprocessing.Process(target=show_camera)
    p.start()
    return p


def translate_text(text, target='en'):
    """Translates text into the target language."""
    result = translate_client.translate(text, target_language=target)
    return result['translatedText']

def start_transcription():
    language_code = "tr-TR"  # Set this as needed
    target_language = "en"   # Set this as needed

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator)

        responses = client.streaming_recognize(streaming_config, requests)

        # Process the transcription responses
        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript
            translation = translate_text(transcript, target=target_language)

            # Update the GUI with transcription and translation
            transcript_var.set(f"{transcript}")
            translation_var.set(f"{translation}")


def text_to_speech(text, language_code="en-US", gender=texttospeech.SsmlVoiceGender.NEUTRAL):
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text)

    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code, 
        ssml_gender=gender
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, 
        voice=voice, 
        audio_config=audio_config
    )

    # Play the audio using pygame
    pygame.mixer.init()
    pygame.mixer.music.load(io.BytesIO(response.audio_content))
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # Wait for audio to finish playing
        pygame.time.Clock().tick(10)

def create_gui():
    global transcript_var, translation_var
    root = tk.Tk()
    root.title("Application with Video Call Feature")

    transcript_var = tk.StringVar()
    translation_var = tk.StringVar()

    tk.Label(root, textvariable=transcript_var).pack()
    tk.Label(root, textvariable=translation_var).pack()

    start_button = tk.Button(root, text="Start Transcription", command=lambda: Thread(target=start_transcription).start())
    start_button.pack()

    # Button to trigger text-to-speech
    tts_button = tk.Button(root, text="Convert Text to Speech", command=lambda: text_to_speech(translation_var.get()))
    tts_button.pack()

    camera_button = tk.Button(root, text="Open Camera", command=run_camera_process)
    camera_button.pack()

    root.mainloop()

if __name__ == '__main__':
    create_gui()
