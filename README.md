# 🚗 Driver Drowsiness Detection System 😴

This project is a real-time Driver Drowsiness Detection system using **OpenCV** and a **Convolutional Neural Network (CNN)**. It alerts the driver with an alarm sound when signs of drowsiness are detected through eye closure patterns.

---

## 🔥 Features
- 👁️ Real-time face and eye detection using Haar cascades
- 🤖 CNN model to detect closed eyes
- 🔊 Alarm sound when drowsiness is detected
- 🖥️ Simple and easy to run on any machine with a webcam

---

## 📦 Requirements

Install the necessary Python libraries using:

```bash
pip install opencv-python tensorflow numpy pygame
```

**Libraries Used:**
- OpenCV (`cv2`)
- TensorFlow / Keras
- NumPy
- Pygame (for playing alarm sound)

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Saihemanth022004/DriverDrowsinessDetection.git
   cd DriverDrowsinessDetection
   ```

2. **Run the main detection script:**
   ```bash
   python "drowsiness detection.py"
   ```

---

## 📂 Project Structure

```
DriverDrowsinessDetection/
│
├── haar cascade files/
│   ├── haarcascade_frontalface_alt.xml
│   ├── haarcascade_lefteye_2splits.xml
│   └── haarcascade_righteye_2splits.xml
│
├── models/
│   └── cnnCat2.h5
│
├── alarm.wav
├── drowsiness detection.py
├── model.py
└── README.md
```
## Screenshots

![drowsy driver](<Screenshot 2025-05-11 085923.png>)
---

## 🎯 How It Works

- The system uses **Haar cascades** to detect the face and eyes in real-time.
- A trained **CNN model** classifies whether the eyes are open or closed.
- If the eyes are detected closed for a certain number of consecutive frames, an **alarm** sound is played to alert the driver.

---

## 🧠 Model Details
- CNN model (`models/cnnCat2.h5`) is trained on an eye dataset.
- Input: Eye image
- Output: Open (0) or Closed (1)

---

## 🙋‍♂️ Author

- 👤 **Sai Hemanth Kumar**
- 🔗 [GitHub Profile](https://github.com/Saihemanth022004)

---

## 🌟 Support

If you find this project useful, consider giving it a ⭐ on GitHub!

---

## 📜 License

This project is open-source and free to use for educational purposes.
