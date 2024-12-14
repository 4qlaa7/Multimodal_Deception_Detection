import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import cv2
import pyaudio
import wave
import threading
import librosa
import mediapipe as mp
import imageio
from itertools import cycle
import logging
from collections import Counter

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load the model
loaded_model = tf.keras.models.load_model('mediapipe_with_audio.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Function to get MFCC features
def get_mfcc_features(audio_path, mfcc_features_list):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_features_list.append(mfcc_mean)

# Function to process video and get facial landmarks
def process_video(video_path, frame_landmarks_list):
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                landmark = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]).flatten()
                frame_landmarks.append(landmark)
        
        # Convert the frame to an image that tkinter can display
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        video_canvas.image = imgtk
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    frame_landmarks_list.extend(frame_landmarks)

# Function to combine features and make predictions
def process_and_predict(video_path):
    try:
        # Extract audio from video
        video_clip = VideoFileClip(video_path)
        if video_clip.audio is None:
            logging.error("No audio track found in the video.")
            return None
        
        audio_path = "temp_audio.wav"
        video_clip.audio.write_audiofile(audio_path)
    except Exception as e:
        logging.error(f"Error extracting audio from video: {e}")
        return None
    
    # Get audio features and video features using threading
    mfcc_features_list = []
    frame_landmarks_list = []

    audio_thread = threading.Thread(target=get_mfcc_features, args=(audio_path, mfcc_features_list))
    video_thread = threading.Thread(target=process_video, args=(video_path, frame_landmarks_list))

    audio_thread.start()
    video_thread.start()

    audio_thread.join()
    video_thread.join()
    
    if not mfcc_features_list or not frame_landmarks_list:
        logging.error("Error: No valid audio or video features were extracted.")
        return None
    
    mfcc_features = mfcc_features_list[0]
    frame_landmarks = frame_landmarks_list
    
    # Combine features
    predictions = []
    for landmark in frame_landmarks:
        if landmark.size > 0:
            combined_features = np.concatenate([mfcc_features, landmark])
            # Reshape the combined features to match the model input shape
            combined_features = combined_features.reshape(1, 1, -1)
            prediction = loaded_model.predict(combined_features)
            predictions.append(prediction)
    
    if predictions:
        predicted_classes = (np.array(predictions) > 0.5).astype(int).flatten()
        logging.info(f"Predicted classes: {predicted_classes}")
        if predicted_classes.size > 0:
            # Count the occurrences of each class
            class_counts = Counter(predicted_classes)
            logging.info(f"Class counts: {class_counts}")
            # Determine the final prediction based on the most frequent class
            final_prediction = class_counts.most_common(1)[0][0]
        else:
            logging.error("Predicted classes array is empty.")
            final_prediction = None
    else:
        logging.error("No valid predictions were made. Check the input video and features.")
        final_prediction = None

    return final_prediction

# Function to handle file selection
def upload_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        display_loading()
        threading.Thread(target=lambda: process_and_display_results(file_path)).start()

# Function to record video with audio
def record_video():
    def start_recording():
        video_path = "temp_video.avi"
        audio_path = "temp_audio.wav"
        combined_path = "combined_output.mp4"
        
        # Set up video recording
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        
        # Set up audio recording
        audio = pyaudio.PyAudio()
        stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        frames = []

        # Record video and audio simultaneously
        for _ in range(200):  # Record for a specific duration
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            data = stream.read(1024)
            frames.append(data)
        
        # Release resources
        cap.release()
        out.release()
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        # Save the audio file
        wf = wave.open(audio_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))
        wf.close()
        
        # Combine video and audio into one file
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        final_clip = video_clip.set_audio(audio_clip)
        final_clip.write_videofile(combined_path, codec='libx264', audio_codec='aac')
        
        display_loading()
        threading.Thread(target=lambda: process_and_display_results(combined_path)).start()
    
    # Start recording in a new thread
    threading.Thread(target=start_recording).start()

# Function to process and display results
def process_and_display_results(file_path):
    final_prediction = process_and_predict(file_path)
    display_results(final_prediction)

# Function to display results
def display_results(final_prediction):
    if final_prediction is not None:
        result_text = f"Final Predicted Class: {'Truthful' if final_prediction == 1 else 'Deceptive'}"
    else:
        result_text = "Error: No valid predictions were made."
    result_label.config(text=result_text)
    stop_loading()

# Function to display loading animation
def display_loading():
    loading_label.pack()
    root.update_idletasks()

def stop_loading():
    loading_label.pack_forget()
    root.update_idletasks()

# Initialize the main window
root = tk.Tk()
root.title("Video Classification with Audio")
root.geometry("800x600")

# Define GUI elements
upload_button = tk.Button(root, text="Upload Video", command=upload_file)
upload_button.pack(pady=20)

record_button = tk.Button(root, text="Record Video", command=record_video)
record_button.pack(pady=20)

result_label = tk.Label(root, text="Results will be displayed here")
result_label.pack(pady=20)

# Load and set up loading animation
loading_gif = imageio.mimread('loading.gif')
loading_images = [ImageTk.PhotoImage(Image.fromarray(img)) for img in loading_gif]
loading_cycle = cycle(loading_images)

loading_label = tk.Label(root)
def update_loading_image():
    loading_label.config(image=next(loading_cycle))
    root.after(100, update_loading_image)

update_loading_image()

# Create a canvas for video display
video_canvas = tk.Canvas(root, width=640, height=480)
video_canvas.pack(pady=20)

# Start the main loop
root.mainloop()

