# BISINDO-YOLO

This repository contains the implementation of **BISINDO (Indonesian Sign Language) hand gesture recognition** using **YOLOv8m** and **YOLOv11m**. The models are trained to detect and classify BISINDO hand gestures, and the system provides a **real-time demonstration** through a **Streamlit** web application using a webcam.

## Features

- Hand gesture recognition for BISINDO
- YOLOv8m and YOLOv11m implementation
- Real-time detection using webcam
- Streamlit-based user interface
- Pretrained model weights included

---

## Repository Structure

```
BISINDO-YOLO/
│
├── main.ipynb              # Main notebook for model development, training, evaluation, and testing
├── app.py                  # Streamlit application for real-time hand gesture recognition
├── best_yolov8m.pt         # Best trained weights for YOLOv8m
├── best_yolov11m.pt        # Best trained weights for YOLOv11m
└── README.md
```

## Running the Streamlit Application

Run the following command:

```bash
streamlit run app.py
```

The application will:

- Open your webcam
- Perform real-time BISINDO gesture detection
- Display the detected gesture label and confidence score

---

## Model Files

The repository includes pretrained weights:

| Model | Weight File |
|--------|-------------|
| YOLOv8m | `best_yolov8m.pt` |
| YOLOv11m | `best_yolov11m.pt` |

These weights are automatically loaded by the Streamlit application.

---

## Notebook

The Jupyter Notebook contains the complete implementation, including:

- Dataset preparation
- Data preprocessing
- Model training
- Validation
- Performance evaluation
- Detection experiments

---

## Technologies

- Python
- Ultralytics YOLO
- PyTorch
- OpenCV
- Streamlit
- NumPy

---

## Citation

If you use this repository in your research, please cite the corresponding thesis.

```
@thesis{,
  title={},
  author={},
  year={2026}
}
```

---

## License

This project is intended for academic and research purposes.
