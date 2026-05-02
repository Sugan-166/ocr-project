# 🚗 Vehicle Number Plate Detection & Recognition System (ANPR)

## 🚀 Overview

This project implements an Automatic Number Plate Recognition (ANPR) system that detects vehicle license plates from images and extracts the alphanumeric text using computer vision and OCR techniques.

The system is designed for real-world applications such as traffic monitoring, toll collection, parking management, and surveillance systems.

---

## 🎯 Features

* 🚘 Detects vehicle number plates from images
* 🔍 Extracts license plate region using computer vision
* 🔤 Recognizes text using OCR
* ⚡ Fast and efficient processing pipeline
* 📊 Outputs structured plate text
* 🧠 Modular and scalable design

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries & Tools:**

  * OpenCV
  * NumPy
  * PIL (Pillow)
  * Tesseract / EasyOCR
* **Concepts Used:**

  * Image preprocessing
  * Contour detection
  * Edge detection
  * OCR (Optical Character Recognition)

---

## 📂 Project Structure

```id="anpr-struct"
ocr-project/
│── data/                  # Input vehicle images
│── outputs/               # Detected plates & extracted text
│── models/                # OCR / detection models (if any)
│
│── detect_text.py         # Plate detection logic
│── ocr_text.py            # Text recognition module
│── ocr_app.py             # Application interface
│── run_ocr_app.py         # Main execution script
│
│── requirements.txt       # Dependencies
│── README.md              # Documentation
```

---

## ⚙️ Installation

1. Clone the repository:

```bash id="anpr-clone"
git clone https://github.com/Sugan-166/ocr-project.git
cd ocr-project
```

2. Create virtual environment:

```bash id="anpr-venv"
python -m venv venv
```

3. Activate environment:

```bash id="anpr-activate"
# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

4. Install dependencies:

```bash id="anpr-install"
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run number plate detection:

```bash id="anpr-run1"
python detect_text.py --input path/to/vehicle_image.jpg
```

### Run full pipeline (Detection + OCR):

```bash id="anpr-run2"
python run_ocr_app.py
```

---

## 📸 Example

**Input:** Vehicle image
**Output:**

```id="anpr-output"
Detected Plate: TN 01 AB 1234
```

(Add sample images here for better impact)

---

## 📈 Applications

* Smart traffic management systems
* Toll booth automation
* Parking access control
* Law enforcement & surveillance
* Vehicle tracking systems

---

## ⚠️ Limitations

* Performance depends on lighting conditions
* Difficulty with blurred or angled plates
* OCR accuracy may vary with font/style

---

## 🔮 Future Improvements

* Real-time video-based detection
* Deep learning-based plate detection (YOLO)
* Improved OCR accuracy using custom-trained models
* Integration with database for vehicle tracking

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork and improve the system.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Sugan**
AI/ML | Computer Vision | Intelligent Systems
GitHub: https://github.com/Sugan-166
