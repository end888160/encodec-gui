# EnCodec Audio Converter GUI (Unofficial)

This is a **Tkinter-based GUI** for **[Facebook Research's EnCodec](https://github.com/facebookresearch/encodec)** audio compression codec. It allows you to:

- Convert audio files to **EnCodec (.ecdc) format**  
- Choose between **24kHz** (low bitrate) and **48kHz** (high-quality) models  
- Adjust the **bitrate** for quality vs. file size trade-off  
- Use **chunking** to process long audio files efficiently  
- Automatically convert non-WAV files using **FFmpeg**  

---

## Installation

### 1. Clone the Repository

Clone the repository from GitHub:

  ```sh
  git clone https://github.com/end888160/encodec-gui.git
  ```

Move into the cloned directory:

  ```sh
  cd encodec-gui
  ```

### 2. Install Dependencies

Make sure you have **Python 3.8+** installed. Then, install the required dependences:

  ```sh
  pip install -r requirements.txt
  ```

### 3. Install FFmpeg (Required for Non-WAV Files)

If you are going to convert MP3, FLAC, or OGG files, install **FFmpeg**:

- **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html) and add it to your system PATH.

- **Linux (Ubuntu/Debian)**:

  ```sh
  sudo apt install ffmpeg
  ```

- **MacOS (Homebrew)**:

  ```sh
  brew install ffmpeg
  ```

### 4. (Optional) Install PyTorch with CUDA for GPU Acceleration

If you have an **NVIDIA GPU**, install PyTorch with CUDA **for much faster processing**:

#### **Check Your CUDA Version**

Run this in **Command Prompt (Windows)** or **Terminal (Linux/Mac):**

  ```sh
  nvcc --version
  ```

If nvcc is not found, try running:

  ```sh
  nvidia-smi
  ```

Look for something like **CUDA Version 12.1**.

#### **Install PyTorch with CUDA**

Replace `cu121` with your **CUDA version**:

  ```sh
  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```

Verify installation:

  ```python
  import torch
  print(torch.cuda.is_available())  # Should return True
  print(torch.cuda.get_device_name(0))  # Should show your GPU name
  ```

---

## Usage

1. **Run the GUI**

  ```sh
  python gui.py
  ```

2. **Select an audio file** (WAV, MP3, FLAC, OGG, etc.)  
3. **Choose the model** (24kHz for small size, 48kHz for high quality)  
4. **Set bitrate** (higher = better quality but larger file size / lower = smaller file size but more artifacty)  
5. **Enable chunking** if converting long audio (prevents out-of-memory errors)  
6. Click **Convert** to encode the file!

---

## Troubleshooting

### 1. **PyTorch CUDA Not Working?**

- Check if CUDA is installed:

  ```sh
  python -c "import torch; print(torch.cuda.is_available())" # Should return True
  ```

- Try reinstalling PyTorch with the correct CUDA version (see installation guide above).

### 2. **FFmpeg Not Found?**

- Make sure it's installed and added to your **system PATH** (Windows users).

### 3. **"RuntimeError: Out of Memory"?**

- Enable **chunking** and increase the **chunk size**.

### 4. **Running out of storage?**

Try:

- Clean up unused files.

- Use a different temporary directory using extra arguments:

  ```sh
  python gui.py --tempdir /path/to/temp/dir
  ```

### 5. **Other Issues?**

- Open an [issue](https://github.com/end888160/encodec-gui/issues) on GitHub.

---

## Links

- **EnCodec GitHub**: [facebookresearch/encodec](https://github.com/facebookresearch/encodec)  
- **PyTorch Installation**: [pytorch.org](https://pytorch.org/get-started/locally/)  

---

## License

This project uses Facebook Research's **EnCodec** under its respective license.

---
