# 🧊 3D Reconstruction from a Single 2D Image

**Digital Image Processing (DIP) + Monocular Depth Estimation → Point Cloud**

An interactive Streamlit app that turns a **single 2D photograph into a 3D point cloud** by combining classical digital image processing techniques with a pre-trained monocular depth estimation model — no stereo camera or LiDAR required.

> 🎓 Built as an academic project exploring the intersection of classical DIP pipelines and modern deep learning-based depth estimation.

🚀 Live Demo

🔗 Try it here (https://nancy-poonia-dip-project.streamlit.app/)

---

## ✨ Features

| Stage | What it does |
|---|---|
| 📤 **Image Upload** | Upload any single RGB image directly in the browser |
| 🧹 **Preprocessing** | Resize, Gaussian denoising, CLAHE contrast enhancement |
| 🌊 **Depth Estimation** | Monocular depth via Hugging Face DPT (`Intel/dpt-hybrid-midas`), with a MiDaS-small `torch.hub` fallback |
| 🛠️ **Depth Refinement** | Median filtering, bilateral filtering, morphological closing |
| ☁️ **Point Cloud Generation** | Depth-to-point-cloud conversion using a pinhole camera model |
| 🎨 **3D Visualization** | Interactive point cloud rendering with Plotly, directly inside Streamlit |

---

## 🧠 How It Works

```
 Input Image
      │
      ▼
 Preprocessing (resize → denoise → CLAHE)
      │
      ▼
 Depth Estimation (DPT / MiDaS)
      │
      ▼
 Depth Post-Processing (median → bilateral → morphological closing)
      │
      ▼
 Pinhole Camera Model → 3D Point Cloud
      │
      ▼
 Interactive Plotly Visualization
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend/App:** Streamlit
- **Depth Model:** Hugging Face `transformers` (DPT) / PyTorch Hub (MiDaS-small)
- **Image Processing:** OpenCV
- **Visualization:** Plotly

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/nancypoonia/DIP-Project.git
cd DIP-Project
```

### 2. Create a virtual environment & install dependencies

**Windows (PowerShell)**
```powershell
python -m venv dip-venv
.\dip-venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS / Linux**
```bash
python -m venv dip-venv
source dip-venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 🔧 Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `DEPTH_BACKEND` | Set to `torchhub` to use the MiDaS-small loader instead of Hugging Face | Hugging Face (DPT) |
| `HF_DEPTH_MODEL_ID` | Override the Hugging Face depth model used | `Intel/dpt-hybrid-midas` |

> ℹ️ On first run, the Hugging Face depth model weights are downloaded automatically (internet connection required). Both backends are CPU-friendly — use the MiDaS fallback if you'd prefer not to install `transformers`.

---

## 📁 Project Structure

```
DIP-Project/
├── app.py                # Main Streamlit application
├── requirements.txt      # Python dependencies
├── assets/                # Static assets used by the app
├── test.jpg               # Sample test image
└── README.md
```

---

## 📚 Academic Notes

This project was built as an exploration of classical digital image processing combined with modern monocular depth estimation. Detailed academic/explanatory notes are included as docstrings throughout the Python modules.

---

## 🗺️ Roadmap / Future Improvements

- [ ] Export point cloud as `.ply` / `.obj`
- [ ] Support batch image processing
- [ ] Add mesh reconstruction from point cloud
- [ ] Deploy live demo (Streamlit Community Cloud)

---

## 👤 Author

**Aadi Jain**
🔗 [GitHub](https://github.com/aadijain558)

