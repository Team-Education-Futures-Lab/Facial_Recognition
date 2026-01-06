# 🧠 Facial_Recognition

An **AI-powered multimodal affect recognition system** that analyzes **facial expressions, tone of voice,** and **gestures,** then transmits results via a custom **Flask server** to a **Unity application** for real-time interaction 🎮.

# 📌 Overview
**Facial_Recognition** provides an **end-to-end affect recognition pipeline** designed for interactive and immersive systems.
By combining **computer vision, audio signal processing,** and **machine learning**, the system interprets human emotions and behavior and exposes the results through a **RESTful API** consumable by Unity.

**💡 Intended Use Cases**
- 📚 Educational tools
- 🧩 Interactive simulations
- 🎮 Games and virtual environments
- 🔬 Research prototypes in affective computing
#

**✨ Features**
- 🎭 Facial Expression Recognition using CNN-based deep learning models
- 🎤 Tone of Voice Analysis via audio feature extraction
- 🧍 Gesture Recognition through video-based tracking
- 🌐 Flask REST API for real-time data streaming
- 🎮 Unity Integration for live emotion and behavior feedback

# 🗂️ Project Structure
```text
Facial_Recognition/
│
├── main.py                  # Flask server entry point
├── train.py                 # Model training script
├── best_model.h5            # Best-performing trained model
├── final_model.h5           # Final production-ready model
├── train/                   # Training dataset
├── validation/              # Validation dataset
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

# 🧠 System Architecture
**🔌 Input Sources**
- 📷 Webcam (facial expressions & gestures)
- 🎙️ Microphone (voice tone)

**⚙️ Processing Pipeline**
- CNN-based facial expression classification
- Audio signal processing for emotional tone analysis
- Computer vision–based gesture detection

**📤 Output**
- Processed affective data served through a Flask API
- Unity retrieves and uses the data in real time
# 🛠️ Installation Guide (Full Setup)

⚠️ Note: The ZIP file is large (~10GB). Make sure you have sufficient disk space and a stable internet connection.

1️⃣ Download & Extract Project

**1.** Download the ZIP file named `AR_Training` (~10GB).

**2.** Extract the ZIP file to your desired location.
#

2️⃣ Unity Installation & Project Setup

**3.** Open **Unity Hub**
- If not installed, download it from: https://unity.com/download
- Update Unity Hub if prompted.

**4.** Click **Add**.

**5.** Navigate to:
`AR_Training → AR_Training → Unity_Project → Open`

**6.** Unity will prompt you to install the correct editor.
Install this editor **with the following build supports:**

- ✅ Android Build Support
- ✅ Universal Windows Platform Build Support
- ✅ Windows Build Support
#
3️⃣ Python & Visual Studio Code Setup

**7.** Open **Visual Studio Code**.

**8.** Click **File → Open Folder**.

**9.** Select the **AR_Training** folder.

**10.** Open the file `StartAI.py`.

**11.** Ensure **Python 3.12** is installed:
- Microsoft Store **or**
- https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

**12.** Install the **Python extension** in VS Code.

**13.** In the bottom-right corner, select the correct Python interpreter.

**14.** Open **Terminal → New Terminal**.

**15.** (Optional) Create a virtual environment.

**16.** Check the Python installation location.
#
4️⃣ Install Python Dependencies

Depending on how Python is installed, use **one** of the following commands
**(replace `{OwnUsername}` with your Windows username):**

**Option A – Standard Python Installation**
```bash
C:\Users\{OwnUsername}\AppData\Local\Programs\Python\Python312\python.exe -m pip install flask requests opencv-python numpy mediapipe tensorflow librosa tqdm scikit-learn joblib matplotlib
```

**Option B – Microsoft Store Python Installation**
```bash
C:\Users\{OwnUsername}\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.PythonManager_{your_python_ID}\python.exe -m pip install flask requests opencv-python numpy mediapipe tensorflow librosa tqdm scikit-learn joblib matplotlib
```


**17.** Ensure the webcam input is set to the **default PC webcam**.

**18.** Wait for installation to finish.

**19.** Press the **Play ▶️ button** on `StartAI.py`.
#
5️⃣ Ollama (LLM Support)

**20.** Download Ollama for Windows:
https://www.ollama.com/download/windows

**21.** Install and open **Ollama**.

**22.** Click the **Model selector** and search for `llama3`.

**23.** Enter any prompt to start downloading the model.

**24.** Wait for the model installation to complete.
#
6️⃣ Database Setup (XAMPP & MySQL)

**25.** Download **XAMPP**:
https://apachefriends.org/download.html

**26.** Install and open the **XAMPP Control Panel**.

**27.** Click **Start** next to:
- Apache
- MySQL

**28.** Open your browser and go to:
http://localhost/phpmyadmin

**29.** Click **Import**.

**30.** Select **Choose File / Browse**.

**31.** Navigate to:
`AR_Training → xampp → 127_0_0_1.sql`

**32.** Click **Import**.
#
7️⃣ Final Unity Launch

**33.** Open **Unity Hub**.

**34.** Click **Unity_Project**

- ⚠️ First launch may show errors; see the **Unity** section for fixes.

**35.** Press **Play ▶️** (top middle of Unity Editor).

**🎉 The system should now be fully operational.**
# 📡 Example API Response
```json
{
  "facial_expression": "happy",
  "expression_confidence": 0.87,
  "voice_tone": "calm",
  "gesture": "wave",
  "timestamp": "2026-01-06T14:23:15Z"
}
```
# 🎮 Unity Integration

Unity communicates with the Flask server using HTTP requests (e.g., `UnityWebRequest`).

**Use Cases in Unity**

🎭 Drive character animations

🎯 Adjust difficulty or behavior dynamically

🧠 Provide affect-aware feedback

📊 Visualize emotional state in real time
# 🧪 Training the Model

To retrain the facial expression model:

```python
python train.py
```


Ensure datasets are placed in:

- `train/`
- `validation/`
# ⚠️ Ethical Considerations

This project processes **biometric and affective data**. Ensure:

Informed user consent

Secure data handling

Awareness of bias and limitations

Compliance with privacy regulations (e.g., GDPR)
# 📄 License

Licensed under the **MIT License**.
See the `LICENSE` file for details.
# 🤝 Contributors

Developed by **Team Education Futures Lab 🌍**
