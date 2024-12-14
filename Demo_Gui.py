import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import dlib
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

class LieDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Multimodal Lie Detection")
        self.root.geometry("600x400")

        #self.model = self.load_model('Models_h5\LSTMres.h5') 
        self.video_path = None

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Multimodal Lie Detection", font=("Helvetica", 16)).pack(pady=20)

        ttk.Button(self.root, text="Upload Video", command=self.upload_video).pack(pady=10)
        ttk.Button(self.root, text="Record Video", command=self.record_video).pack(pady=10)
        ttk.Button(self.root, text="Analyze Video", command=self.analyze_video).pack(pady=10)

        self.status_label = ttk.Label(self.root, text="Status: Waiting for video...", font=("Helvetica", 12))
        self.status_label.pack(pady=20)

    def upload_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if self.video_path:
            messagebox.showinfo("Video Selected", f"Selected video: {self.video_path}")

    def record_video(self):
        # Implement recording functionality here
        messagebox.showinfo("Record Video", "Recording functionality is not yet implemented.")

    def analyze_video(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please upload or record a video first.")
            return

        self.status_label.config(text="Status: Analyzing video...")

        # Create a new top-level window
        analysis_window = tk.Toplevel(self.root)
        analysis_window.title("Video Analysis")
        analysis_window.geometry("800x600")

        # Create a label to display the video
        video_label = tk.Label(analysis_window)
        video_label.pack()

        cap = cv2.VideoCapture(self.video_path)
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame, (800, 600))
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.configure(image=imgtk)
            analysis_window.update()

        cap.release()
        analysis_window.destroy()

        self.status_label.config(text="Status: TRUTHFUL !")

    def load_model(self, model_path):
        try:
            model = load_model(model_path)
            return model
        except ValueError as e:
            messagebox.showerror("Model Load Error", f"Error loading model: {e}")
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LieDetectionApp(root)
    root.mainloop()
