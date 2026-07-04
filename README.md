# BISINDO-YOLO

This repository contains the implementation of **BISINDO (Indonesian Sign Language) hand gesture recognition** using **YOLOv8m** and **YOLOv11m**. The models are trained to detect and classify BISINDO hand gestures, and the system provides a **real-time demonstration** through a **Streamlit** web application using a webcam.

---

## Repository Structure

```text
BISINDO-YOLO/
│
├── main.ipynb          # Main notebook for dataset preparation, training, validation, evaluation, and testing
├── bisindo_app.py      # Streamlit application for real-time BISINDO gesture recognition
├── requirements.txt    # Python dependencies
└── README.md
```

---

## Requirements

Install all required packages:

```bash
pip install -r requirements.txt
```

---

## Getting Started

### Step 1. Train the Model

Open and run **`main.ipynb`** to:

- Prepare the dataset
- Train either the YOLOv8m or YOLOv11m model
- Validate and evaluate the trained model
- Generate the best model weights (`best.pt`)

> **Note:** The pretrained model weights are **not included** in this repository. You must run the notebook first to generate the `best.pt` file before using the Streamlit application.

---

### Step 2. Run the Streamlit Application

After obtaining the trained `best.pt` model, launch the real-time application:

```bash
streamlit run bisindo_app.py
```

The application will:

- Open the webcam
- Detect BISINDO hand gestures in real time
- Display the predicted gesture label and confidence score

---

## License

This project is intended for academic and research purposes.
