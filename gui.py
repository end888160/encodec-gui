import sys
print("Python version:", sys.version)

import os
import time
import zipfile
import logging
import shutil
import tempfile
import threading
import traceback
import webbrowser
import subprocess
from tqdm import tqdm

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import torch
import soundfile
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
print("PyTorch version:", torch.__version__)

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

# GUI Setup
root = tk.Tk()
root.title("EnCodec Audio Converter")

ui_width = 240
ui_height = 480
root.geometry(f"{ui_width}x{ui_height}")

selected_file = ""
selected_model = tk.StringVar(value="48kHz")
selected_bitrate = tk.StringVar(value="6.0")
use_chunking_var = tk.BooleanVar(value=True)

class ToolTip:
	def __init__(self, widget, text):
		self.widget = widget
		self.text = text
		self.tooltip_window = None

		# Bind mouse events
		self.widget.bind("<Enter>", self.show_tooltip)
		self.widget.bind("<Leave>", self.hide_tooltip)

	def show_tooltip(self, event=None):
		"""Creates and shows tooltip window"""
		if self.tooltip_window:
			return
		
		x, y, _, _ = self.widget.bbox("insert")
		x += self.widget.winfo_rootx() + 25
		y += self.widget.winfo_rooty() + 25

		# Create Toplevel window for tooltip
		self.tooltip_window = tk.Toplevel(self.widget)
		self.tooltip_window.wm_overrideredirect(True)  # No window decorations
		self.tooltip_window.wm_geometry(f"+{x}+{y}")

		# Create Label inside the tooltip window
		label = tk.Label(self.tooltip_window, text=self.text, bg="white", fg="black", relief="solid", borderwidth=1, padx=5, pady=2)
		label.pack()

	def hide_tooltip(self, event=None):
		"""Hides tooltip window"""
		if self.tooltip_window:
			self.tooltip_window.destroy()
			self.tooltip_window = None

def is_tool(name: str) -> bool:
	"""Check whether `name` is on PATH and marked as executable."""

	# from whichcraft import which
	from shutil import which

	return which(name) is not None


def open_git_repository(event):
	webbrowser.open("https://github.com/facebookresearch/encodec")

def choose_file():
	global selected_file
	selected_file = filedialog.askopenfilename()
	if selected_file:
		file_input.delete(0, tk.END)
		file_input.insert(0, selected_file)

def choose_output_folder():
	global output_folder
	output_folder = filedialog.askdirectory()
	if output_folder:
		folder_output.delete(0, tk.END)
		folder_output.insert(0, output_folder)



# Preload models
# Check CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device.upper()}")
models = {}
try:
	logging.info("Loading 24kHz EnCodec model...")
	models[24] = EncodecModel.encodec_model_24khz().to(device)
except Exception as e:
	logging.error(f"Error loading EnCodec model: {e}")

try:
	logging.info("Loading 48kHz EnCodec model...")
	models[48] = EncodecModel.encodec_model_48khz().to(device)
except Exception as e:
	logging.error(f"Error loading EnCodec model: {e}")

def recompress_zip(input_zip: str, output_zip: str, compression=zipfile.ZIP_DEFLATED, compression_level=9):
    """
    Recompress a ZIP file at a higher compression level.
    
    :param input_zip: Path to the input ZIP file.
    :param output_zip: Path to the output recompressed ZIP file.
    """
    temp_dir = "temp_extracted_zip"
    
    # Ensure temp directory is clean
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    try:
        # Extract existing ZIP
        with zipfile.ZipFile(input_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Recompress with higher compression
        with zipfile.ZipFile(output_zip, 'w', compression=compression, compresslevel=compression_level) as zip_out:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zip_out.write(file_path, arcname)
    
    finally:
        # Cleanup temporary directory
        shutil.rmtree(temp_dir)

	
def encode_audio_thread():
	global file_input, output_folder, use_chunking_var, task_ended, seconds_per_chunk, input_file, output_file
	task_ended = False
	start_time = time.time()
	try:
		logging.info(f"Encoding '{input_file}'...")
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		if not input_file.endswith(".wav"):
			logging.info("Converting to WAV...")
			if is_tool("ffmpeg"):
				status_label.config(text="Converting to WAV...")
				root.update_idletasks()

				temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
				sample_rate = 48000 if selected_model.get() == "48kHz" else 24000
				cmd = ["ffmpeg", "-i", input_file, "-ar", str(sample_rate),"-y", temp_wav_file.name]
				print(" ".join(cmd))
				subprocess.run(cmd, check=True)
				input_file = temp_wav_file.name
			else:
				logging.error("FFmpeg is not installed. Please install it and try again.")
				messagebox.showerror("Error", "FFmpeg is not installed. Please install it and try again.")
				task_ended = True
				return
		model = models[48] if selected_model.get() == "48kHz" else models[24]
		model.set_target_bandwidth(float(selected_bitrate.get()))

		# Show progress bar
		
		status_label.config(text="Encoding...")
		root.update_idletasks()

		if not use_chunking_var.get():
			# Encode audio
			logging.info("Encoding audio...")
			wav, sr = torchaudio.load(input_file)
			logging.info(f"Loaded audio with shape {wav.shape} and sample rate {sr}")
			wav = convert_audio(wav, sr, model.sample_rate, model.channels)
			logging.info(f"Converted audio with shape {wav.shape} and sample rate {model.sample_rate}")
			wav = wav.unsqueeze(0).to(device)
			progress_bar["maximum"] = wav.shape[1]
			root.update_idletasks()
			logging.info("Encoding audio...")
			with torch.no_grad():
				encoded_frames = model.encode(wav)
			logging.info("Encoded audio")
			torch.save(encoded_frames, output_file)
			logging.info("Recompressing...")
			recompress_zip(output_file, output_file)
			progress_bar["value"] = wav.shape[1]
			root.update_idletasks()

		else:
			
			logging.info(f"Encoding audio in chunks of {seconds_per_chunk} seconds")
			chunk_size = int(model.sample_rate * float(seconds_per_chunk))
			# Load audio
			logging.info("Loading audio...")
			wav, sr = torchaudio.load(input_file)
			logging.info(f"Loaded audio with shape {wav.shape} and sample rate {sr}")
			wav = convert_audio(wav, sr, model.sample_rate, model.channels).to(device)

			# Process in smaller chunks (e.g., 10 seconds per chunk)
			encoded_chunks = []
			progress_bar["maximum"] = wav.shape[1]
			logging.info("Encoding audio in chunks...")
			with torch.no_grad():
				# Show progress bar
				dyn_tqdm = tqdm(range(0, wav.shape[1], chunk_size), desc="Encoding chunks", unit="chunk")
				for i in dyn_tqdm:
					chunk = wav[:, i : i + chunk_size].unsqueeze(0)  # Add batch dim
					encoded_chunks.append(model.encode(chunk))
					# Update progress bar
					progress_bar["value"] = i + chunk_size
					processed_duration = i / model.sample_rate
					if processed_duration > 86400:
						processed_duration /= 86400
						time_unit = "day"
					elif processed_duration > 3600:
						processed_duration /= 3600
						time_unit = "hr."
					elif processed_duration > 60:
						processed_duration /= 60
						time_unit = "min."
					else:
						time_unit = "sec."
					status_label.config(text=f"Encoding... {i / wav.shape[1] * 100:.2f}% ({processed_duration:.2f} {time_unit})")
					dyn_tqdm.set_postfix({"Time": f"{i / model.sample_rate:.2f} sec."})
					root.update_idletasks()

			# Save all encoded chunks
			status_label.config(text="Saving encoded chunks...")
			root.update_idletasks()
			logging.info("Saving encoded chunks...")
			torch.save(encoded_chunks, output_file)
			logging.info("Recompressing...")
			recompress_zip(output_file, output_file)
		status_label.config(text="Done!")
		root.update_idletasks()
		logging.info("Encoding complete! Saved as " + output_file)
		messagebox.showinfo("Success", "Encoding complete! Saved as " + output_file + "\nElapsed time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))
		progress_bar["value"] = 0
	except Exception as e:
		print(traceback.format_exc())
		logging.error(f"Failed to encode '{input_file}' due to {e}")
		messagebox.showerror(type(e).__name__, traceback.format_exc())
	finally:
		status_label.config(text="Idle")
		root.update_idletasks()
		task_ended = True

task_ended = True
def encode_audio():
	# Don't allow multiple encodes to run at the same time
	if task_ended == False:
		messagebox.showerror("Error", "Only one encoding process can run at a time.")
		return
	if not file_input.get():
		messagebox.showerror("Error", "No file selected")
		return
	if selected_bitrate.get() == "1.5" and selected_model.get() == "48kHz":
		messagebox.showerror("Error", "1.5 kbps with 48khz model is not supported")
		return
	if not os.path.isfile(file_input.get()):
		messagebox.showerror("Error", "File not found")
		return
	global seconds_per_chunk, input_file, output_file, output_folder
	try:
		seconds_per_chunk = float(chunk_length_entry.get())
		if seconds_per_chunk <= 0:
			messagebox.showerror("Error", "Chunk length must be greater than 0")
	except ValueError:
		messagebox.showerror("Error", "Invalid chunk length")
		return
	output_folder = folder_output.get()
	input_file = file_input.get()
	output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".ecdc")
	if os.path.exists(output_file):
		ask_overwrite = messagebox.askyesno("File already exists", "File already exists. Do you want to overwrite it?")
		if not ask_overwrite:
			return
	global encoding_thread
	encoding_thread = threading.Thread(target=encode_audio_thread, daemon=True)
	encoding_thread.start()

'''def abort_encoding():
	global encoding_thread
	encoding_thread.join()
	encoding_thread = None'''

device_label = tk.Label(root, text=f"Running on {device.upper()}")	
device_label.pack()


# UI Elements
file_input = tk.Entry(root, width=40)
file_input.pack(pady=10, padx=4)

btn_choose = tk.Button(root, text="Choose File", command=choose_file, width=15)
btn_choose.pack()

folder_output = tk.Entry(root, width=40)
folder_output.pack(pady=10, padx=4)

folder_output.insert(0, "./output")

btn_output = tk.Button(root, text="Choose Output Folder", command=choose_output_folder, width=20)
btn_output.pack()


model_label = tk.Label(root, text="Select Model")
model_label.pack()

ToolTip(model_label, "24kHz: Lower sample rate, good for speech and small file sizes. Supports bitrates: 1.5, 3, 6, 12 kbps.\n48kHz: High-quality mode, best for music. Supports bitrates: 3, 6, 12, 24 kbps.")

tk.Radiobutton(root, text="24kHz", variable=selected_model, value="24kHz").pack()
tk.Radiobutton(root, text="48kHz", variable=selected_model, value="48kHz").pack()

bitrate_label = tk.Label(root, text="Bitrate (kbps)")
bitrate_label.pack()

tk.OptionMenu(root, selected_bitrate, "1.5", "3.0", "6.0", "12.0", "24.0").pack()

ToolTip(bitrate_label, "Higher bitrates improve quality but increase file size. Lower bitrates (e.g., 1.5 kbps) may introduce robotic-sounding artifacts.") 

use_chunking = tk.Checkbutton(root, text="Use Chunking", variable=use_chunking_var)
use_chunking.pack()

ToolTip(use_chunking, "Chunking prevents high RAM usage for long audio files. If you hear clicking sounds, try increasing chunk size or disabling chunking.")

chunk_size_label = tk.Label(root, text="Chunk Size (seconds)")
chunk_size_label.pack()

ToolTip(chunk_size_label, "Defines how long each chunk should be when processing audio. A larger chunk size reduces clicking but increases memory usage.")

chunk_length_entry = tk.Entry(root)	
chunk_length_entry.insert(0, "10")
chunk_length_entry.pack()

btn_encode = tk.Button(root, text="Convert", command=encode_audio)
btn_encode.pack(pady=5)

# TODO: Add abort button
'''btn_abort = tk.Button(root, text="Abort", command=abort_encoding)
btn_abort.pack(pady=5)'''

status_label = tk.Label(root, text="Idle")	
status_label.pack()

progress_bar = ttk.Progressbar(root, length=root.winfo_screenwidth() - 20, mode="determinate")
progress_bar.pack(pady=10, padx=4)

link_label = tk.Label(root, text="GitHub", fg="blue", cursor="hand2")
link_label.pack()

link_label.bind("<Button-1>", open_git_repository)	

ToolTip(link_label, "Click to visit the official Facebook Research EnCodec repository on GitHub for more details.")

root.mainloop()
