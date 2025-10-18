#!./venv/bin/python
import sys
print("Python version:", sys.version)

import os
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/SSPL-UT/encodec/models/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/mnt/SSPL-UT/encodec/models/transformers"

import argparse
import logging

logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")
parser = argparse.ArgumentParser(description='EnCodec Audio Converter GUI', epilog='There are no other arguments because this is a GUI')
parser.add_argument('-t', '--tmpdir', type=str, default=None, help='temporary directory')
parser.add_argument('-d','--device', type=str, default=None, help='device to use (cpu or cuda)')
args = parser.parse_args()


import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox

import time
import shutil
import zipfile
import tempfile
import threading
import traceback
import webbrowser
import subprocess
from tqdm import tqdm
import soundfile as sf
import numpy as np
TEMP = args.tmpdir if args.tmpdir else tempfile.gettempdir()
logging.info(f"Using temporary directory: {TEMP}")

import torch
import soundfile
import torchaudio
from encodec import EncodecModel
from encodec.utils import convert_audio
print("PyTorch version:", torch.__version__)



# GUI Setup
root = tk.Tk()
root.title("EnCodec Audio Converter")

ui_width = 240
ui_height = 540
root.geometry(f"{ui_width}x{ui_height}")

selected_file = ""
selected_model = tk.StringVar(value="48kHz")
selected_model2 = tk.StringVar(value="48kHz")
selected_bitrate = tk.StringVar(value="6.0")
convert_format = tk.StringVar(value="wav")
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

def choose_file(file_input):
	global selected_file
	selected_file = filedialog.askopenfilename()
	if selected_file:
		file_input.delete(0, tk.END)
		file_input.insert(0, selected_file)

def choose_output_folder(folder_output):
	global output_folder
	output_folder = filedialog.askdirectory()
	if output_folder:
		folder_output.delete(0, tk.END)
		folder_output.insert(0, output_folder)

def get_free_disk_space(path):
	# Get free disk space in bytes
	free_space = shutil.disk_usage(path).free
	return free_space


# Preload models
# Check CUDA
device = args.device if args.device else "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device.upper()}")
models = {}
try:
	models[24] = EncodecModel.encodec_model_24khz().to(device)
	logging.info("Loaded 24kHz EnCodec model")
except Exception as e:
	logging.error(f"Error loading 24kHz EnCodec model: {e}")

try:
	models[48] = EncodecModel.encodec_model_48khz().to(device)
	logging.info("Loaded 48kHz EnCodec model")
except Exception as e:
	logging.error(f"Error loading 48kHz EnCodec model: {e}")

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

def short_time(s):
	if s > 86400:
		s /= 86400
		unit = "day"
	elif s > 3600:
		s /= 3600
		unit = "hr."
	elif s > 60:
		s /= 60
		unit = "min."
	else:
		unit = "sec."
		
	return f"{s:.2f} {unit}"
	
def encode_audio_thread():
	global file_input, output_folder, encode_abort, use_chunking_var, task_ended, seconds_per_chunk, input_file, output_file
	task_ended = False
	start_time = time.time()
	try:
		# Get original audio duration
		ffprobe_cmd = [
			"ffprobe",
			"-v", "quiet",
			"-show_entries", "format=duration",
			"-of", "default=noprint_wrappers=1:nokey=1",
			input_file
		]
		original_audio_duration = subprocess.check_output(ffprobe_cmd).decode().strip()
		original_audio_duration = float(original_audio_duration)
		logging.info(f"Original audio duration: {original_audio_duration:.2f} seconds")
		logging.info(f"Encoding '{input_file}'...")
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		if not input_file.endswith(".wav"):
			logging.info("Converting to WAV...")
			if is_tool("ffmpeg"):
				status_label.config(text="Converting to WAV...")
				root.update_idletasks()
				temp_wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=TEMP)
				sample_rate = 48000 if selected_model.get() == "48kHz" else 24000
				cmd = [
					"ffmpeg", 
					'-hide_banner',
					"-i", input_file, 
					"-ar", str(sample_rate),
					"-y", 
					temp_wav_file.name
				]
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
				pbar = tqdm(range(0, wav.shape[1], chunk_size), desc="Encoding chunks", unit="chunk")
				for i in pbar:
					if encode_abort:
						logging.info("Encoding aborted by user.")
						status_label.config(text="Aborted.")
						encode_abort = False
						btn_encode.config(text="Convert", command=encode_audio, state="normal")
						root.update_idletasks()
						return
					chunk = wav[:, i : i + chunk_size].unsqueeze(0)  # Add batch dim
					encoded_chunks.append(model.encode(chunk))
					# Update progress bar
					progress_bar["value"] = i + chunk_size
					processed_duration = i / model.sample_rate
					
					status_label.config(text=f"Encoding... {i / wav.shape[1] * 100:.2f}% ({short_time(processed_duration)})")
					pbar.set_postfix({"Time": f"{i / model.sample_rate:.2f} sec."})
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
		output_size = os.path.getsize(output_file)
		output_bitrate_kbps = (output_size * 8) / original_audio_duration / 1000
		logging.info("Encoding complete! Saved as " + output_file)
		messagebox.showinfo("Success", "Encoding complete! Saved as " + output_file + "\nElapsed time: " + time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)) + f"\nOutput size: {output_size / 1024:.2f} KB ({output_bitrate_kbps:.2f} kbps)")
		progress_bar["value"] = 0
	except Exception as e:
		status_label.config(text="Error")
		root.update_idletasks()
		print(traceback.format_exc())
		logging.error(f"Failed to encode '{input_file}' due to {e}")
		messagebox.showerror(type(e).__name__, traceback.format_exc())
	finally:
		if os.path.exists(temp_wav_file.name):
			os.remove(temp_wav_file.name)
		btn_encode.config(text="Convert", command=encode_audio)
		btn_encode.update_idletasks()
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
	disk_space = get_free_disk_space(TEMP)
	logging.info(f"Free disk space: {disk_space / 1024 ** 2:.2f} MB")
	if disk_space < 1024 ** 2:
		messagebox.showerror("Disk is full", f"Not enough disk space ({disk_space / 1024 ** 2:.2f} MB)\nPlease choose a different temp folder.")
		return
	if disk_space < 1024 ** 3:
		ask_continue = messagebox.askyesno("Low disk space", f"Low disk space. ({disk_space / 1024 ** 2:.2f} MB)\nDo you want to continue?")
		if not ask_continue:
			return
	output_folder = folder_output.get()
	input_file = file_input.get()
	output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + ".ecdc")
	if os.path.exists(output_file):
		ask_overwrite = messagebox.askyesno("File already exists", "File already exists. Do you want to overwrite it?")
		if not ask_overwrite:
			return
	btn_encode.config(text="Abort", command=abort_encode)
	global encoding_thread
	encoding_thread = threading.Thread(target=encode_audio_thread, daemon=True)
	encoding_thread.start()

def concatenate_wavs(part_files, output_file):
    """Fully stream concatenate (safe for mono or stereo)."""

    first = True
    with sf.SoundFile(output_file, mode='w', samplerate=44100, channels=1) as f_out:
        for f in part_files:
            data, sr = sf.read(f, dtype='float32')
            if data.ndim == 1:  # mono
                data = data[:, np.newaxis]

            if first:
                f_out.close()
                f_out = sf.SoundFile(output_file, mode='w', samplerate=sr, channels=data.shape[1])
                first = False

            f_out.write(data)



def decode_audio_thread():
	global decode_abort
	failed = False
	try:
		status_label.config(text="Decoding...")
		root.update_idletasks()
		start_time = time.time()

		input_file = file_input2.get()
		output_file = os.path.splitext(input_file)[0] + f".wav"
		if os.path.exists(output_file):
			ask_overwrite = messagebox.askyesno("File already exists", "File already exists. Do you want to overwrite it?")
			if not ask_overwrite:
				return

		model = models[48] if selected_model2.get() == "48kHz" else models[24]
		output_base = os.path.splitext(input_file)[0]
		with open(input_file, "rb") as f:
			encoded_chunks = torch.load(f)

		processed_duration = 0
		progress_bar["maximum"] = len(encoded_chunks)
		root.update_idletasks()

		# if not chunked file
		part_files = []

		pbar = tqdm(encoded_chunks, desc="Decoding chunks", unit="chunk")
		for i, chunk in enumerate(pbar):
			with torch.no_grad():
				decoded_chunk = model.decode(chunk)[0].cpu()
			if decode_abort:
				logging.info("Decoding aborted by user.")
				status_label.config(text="Aborted.")
				decode_abort = False
				btn_decode.config(text="Convert", command=decode_audio, state="normal")
				root.update_idletasks()
				return

			part_file = os.path.join(TEMP, f"part_{i:05d}.wav")
			torchaudio.save(part_file, decoded_chunk, model.sample_rate)
			part_files.append(part_file)
			processed_duration += decoded_chunk.shape[1] / model.sample_rate

			del decoded_chunk, chunk
			torch.cuda.empty_cache()

			pbar.set_postfix({"Time": f"{processed_duration:.2f} sec."})
			status_label.config(text=f"Decoding... {(i + 1) / len(encoded_chunks) * 100:.2f}% ({short_time(processed_duration)})")
			progress_bar["value"] = i + 1
			root.update_idletasks()

		if decode_abort:
			logging.info("Decoding aborted by user.")
			status_label.config(text="Aborted.")
			decode_abort = False
			btn_decode.config(text="Convert", command=decode_audio, state="normal")
			root.update_idletasks()
			return

		# Concatenate all parts
		status_label.config(text="Concatenating WAV files...")
		root.update_idletasks()
		final_wav = f"{output_base}.wav"

		concatenate_wavs(part_files, final_wav)

		# Clean up TEMP if desired
		# import shutil; shutil.rmtree(temp_dir)
	except Exception:
		try: # Try non-chunked decode if chunked decode fails
			logging.info("Chunked decode failed, trying non-chunked decode...")
			status_label.config(text="Decoding (non-chunked)...")
			root.update_idletasks()
			start_time = time.time()

			input_file = file_input2.get()
			output_file = os.path.splitext(input_file)[0] + ".wav"

			model = models[48] if selected_model2.get() == "48kHz" else models[24]
			with open(input_file, "rb") as f:
				encoded_frames = torch.load(f, map_location=device)

			status_label.config(text="Decoding...")
			root.update_idletasks()
			logging.info("Decoding audio...")
			with torch.no_grad():
				wav = model.decode(encoded_frames)[0].cpu().numpy()
			torchaudio.save(output_file, torch.tensor(wav), model.sample_rate, model.channels)
			root.update_idletasks()

		except Exception as e:
			failed = True
			status_label.config(text="Error")
			root.update_idletasks()
			print(traceback.format_exc())
			logging.error(f"Failed to decode '{input_file}' due to {e}")
			messagebox.showerror(type(e).__name__, traceback.format_exc())

	if convert_format and not failed and convert_format.get() != "wav":
		# Convert to desired format if not WAV
		status_label.config(text=f"Converting to {convert_format.get()}...")
		root.update_idletasks()
		output_file_conv = os.path.splitext(output_file)[0] + f".{convert_format.get()}"
		try:
			if is_tool("ffmpeg"):
				cmd = [
					"ffmpeg", 
					'-hide_banner',
					"-i", output_file, 
					"-y", 
					output_file_conv
				]
				print(" ".join(cmd))
				subprocess.run(cmd, check=True)
				if os.path.exists(output_file_conv):
					os.remove(output_file)  # Remove original WAV
				output_file = output_file_conv
				status_label.config(text="Done!")
				root.update_idletasks()
				logging.info(f"Converted to {output_file}")
			else:
				logging.error("FFmpeg is not installed. Please install it and try again.")
				messagebox.showerror("Error", "FFmpeg is not installed. Please install it and try again.")
			output_file = output_file_conv
		except Exception as e:
			failed = True
			status_label.config(text="Error")
			root.update_idletasks()
			print(traceback.format_exc())
			logging.error(f"Failed to convert '{output_file}' due to {e}")
			messagebox.showerror(type(e).__name__, traceback.format_exc())
	
	if not failed and os.path.exists(output_file):
		status_label.config(text="Done!")
		btn_decode.config(text="Convert", command=decode_audio)
		btn_decode.update_idletasks()
		elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
		logging.info(f"Decoding complete! Saved as {output_file}, took {elapsed}")
		messagebox.showinfo("Success", f"Decoding complete! Saved as {output_file}\nElapsed: {elapsed}")	
		progress_bar["value"] = 0
		root.update_idletasks()
		status_label.config(text="Idle")

def decode_audio():
	if not file_input2.get():
		messagebox.showerror("Error", "No file selected")
		return
	if not os.path.isfile(file_input2.get()):
		messagebox.showerror("Error", "File not found")
		return
	btn_decode.config(text="Abort", command=abort_decode)
	global decoding_thread
	decoding_thread = threading.Thread(target=decode_audio_thread, daemon=True)
	decoding_thread.start()

def abort_encode():
    global encode_abort
    encode_abort = True
    status_label.config(text="Aborting...")
    btn_encode.config(state="disabled")  # temporarily disable to prevent spamming
    root.update_idletasks()


def abort_decode():
    global decode_abort
    decode_abort = True
    status_label.config(text="Aborting...")
    btn_decode.config(state="disabled")
    root.update_idletasks()



device_label = tk.Label(root, text=f"Running on {device.upper()}")	
device_label.pack()

# UI Elements

notebook = ttk.Notebook(root)

notebook.pack(expand=True, fill="both")
frame_encode = tk.Frame(notebook)
frame_decode = tk.Frame(notebook)

notebook.add(frame_encode, text="Encode")
notebook.add(frame_decode, text="Decode")

file_input = tk.Entry(frame_encode, width=40)
file_input2 = tk.Entry(frame_decode, width=40)
file_input.pack(pady=10, padx=4)
file_input2.pack(pady=10, padx=4)

btn_choose = tk.Button(frame_encode, text="Choose File", command=lambda: choose_file(file_input), width=15)
btn_choose2 = tk.Button(frame_decode, text="Choose File", command=lambda: choose_file(file_input2),  width=15)
btn_choose.pack()
btn_choose2.pack()

folder_output = tk.Entry(frame_encode, width=40)
folder_output2 = tk.Entry(frame_decode, width=40)
folder_output.pack(pady=10, padx=4)
folder_output2.pack(pady=10, padx=4)

folder_output.insert(0, "./output")
folder_output2.insert(0, "./output")

btn_output = tk.Button(frame_encode, text="Choose Output Folder", command=lambda: choose_output_folder(folder_output), width=20)
btn_output2 = tk.Button(frame_decode, text="Choose Output Folder", command=lambda: choose_output_folder(folder_output2), width=20)
btn_output.pack()
btn_output2.pack()


model_label = tk.Label(frame_encode, text="Select Model")
model_label.pack()

ToolTip(model_label, "24kHz: Lower sample rate, good for speech and small file sizes. Supports bitrates: 1.5, 3, 6, 12 kbps.\n48kHz: High-quality mode, best for music. Supports bitrates: 3, 6, 12, 24 kbps.")
tk.Radiobutton(frame_encode, text="48kHz", variable=selected_model, value="48kHz").pack()
tk.Radiobutton(frame_encode, text="24kHz", variable=selected_model, value="24kHz").pack()

model_label2 = tk.Label(frame_decode, text="Select Model")
model_label2.pack()
ToolTip(model_label2, "Choose the model to use for decoding. Tip: try 24kHz first, then switch to 48kHz if you get errors.")

tk.Radiobutton(frame_decode, text="48kHz", variable=selected_model2, value="48kHz").pack()
tk.Radiobutton(frame_decode, text="24kHz", variable=selected_model2, value="24kHz").pack()

bitrate_label = tk.Label(frame_encode, text="Bitrate (kbps)")
bitrate_label.pack()

tk.OptionMenu(frame_encode, selected_bitrate, "1.5", "3.0", "6.0", "12.0", "24.0").pack()

ToolTip(bitrate_label, "Higher bitrates improve quality but increase file size. Lower bitrates (e.g., 1.5 kbps) may introduce robotic-sounding artifacts.") 

use_chunking = tk.Checkbutton(frame_encode, text="Use Chunking", variable=use_chunking_var)
use_chunking.pack()

ToolTip(use_chunking, "Chunking prevents high RAM usage for long audio files. If you hear clicking sounds, try increasing chunk size or disabling chunking.")

chunk_size_label = tk.Label(frame_encode, text="Chunk Size (seconds)")
chunk_size_label.pack()

ToolTip(chunk_size_label, "Defines how long each chunk should be when processing audio. A larger chunk size reduces clicking but increases memory usage.")

chunk_length_entry = tk.Entry(frame_encode, width=5)	
chunk_length_entry.insert(0, "10")
chunk_length_entry.pack()

convert_format_label = tk.Label(frame_decode, text="Convert Format")
convert_format_label.pack()
tk.OptionMenu(frame_decode, convert_format, "wav", "flac", "ogg", "mp3", "aac").pack()
ToolTip(convert_format_label, "Select the output format after decoding. Requires FFmpeg to be installed.")


btn_encode = tk.Button(frame_encode, text="Convert", command=encode_audio)
btn_decode = tk.Button(frame_decode, text="Convert", command=decode_audio)
btn_encode.pack(pady=5)
btn_decode.pack(pady=5)

encode_abort = False
decode_abort = False

# TODO: Add abort button
'''btn_abort = tk.Button(frame_encode, text="Abort", command=abort_encoding)
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
