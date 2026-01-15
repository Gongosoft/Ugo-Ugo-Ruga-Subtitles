import os
import re
import sys
import shutil
import logging
import warnings
import argparse
from yt_dlp import YoutubeDL
from argparse import ArgumentDefaultsHelpFormatter

warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pyannote.*')

from pyannote.audio.utils.reproducibility import ReproducibilityWarning

logging.getLogger('speechbrain.utils.quirks').setLevel(logging.WARNING)
logging.getLogger('pytorch_lightning.utilities.migration.utils').setLevel(logging.WARNING)

import whisperx
from whisperx.utils import get_writer

def get_subtitles(urls: list, cookies_path: str, subs_path: str, whisper_model: str, whisper_compute_type: str, whisper_batch_size: int, whisper_device: str):
	class UnavailableSubsLogger:
		def __init__(self):
			self.unavailable_log_file = open('unavailable_videos.txt', 'a', encoding='utf-8')
			self.current_video_id = ''

		def debug(self, msg):
			if '[download]' not in msg:
				print(msg, flush=True)
			elif '[download]' in msg:
				print(f'\r{msg}', end='', flush=True)
			if '[youtube] Extracting URL: https://www.youtube.com/watch?v=' in msg:
				self.current_video_id = msg.split('[youtube] Extracting URL: https://www.youtube.com/watch?v=')[1]

		def warning(self, msg):
			print(msg)

		def error(self, msg):
			print(msg, file=sys.stderr)
			if 'Video unavailable.' in msg:
				self.unavailable_log_file.write(f'https://www.youtube.com/watch?v={self.current_video_id}\n')
				with open('downloaded.txt', 'a', encoding='utf-8') as f:
					f.write(f'youtube {self.current_video_id}\n')

		def close(self):
			self.unavailable_log_file.close()

	logger = UnavailableSubsLogger()

	args = {
		'format': 'bestaudio/best',
		'cookiefile': cookies_path,
		'download_archive': 'downloaded.txt',
		'force_write_download_archive': True,
  		'ignoreerrors': 'only_download',
		'js_runtimes': {'bun': {}},
		'ignoreerrors': False,
		'logger': logger,
		'outtmpl': '%(title)s.%(ext)s',
		'paths': {'home': subs_path},
		'remote_components': ['ejs:npm'],
		'restrictfilenames': False,
		'simulate': False
	}

	os.makedirs(subs_path, exist_ok=True)

	with YoutubeDL(args) as ydl:
		ydl.download(urls)

		subtitle_files = [f for f in os.listdir(subs_path) if f.endswith('.srt')]
		pattern = re.compile(r'^(?P<title>.+?)\.en(?:-[^.]+)?\.srt$')
		files_by_title = {}

		for file in subtitle_files:
			if match := pattern.match(file):
				files_by_title.setdefault(match.group('title'), []).append(file)

		for title, files in files_by_title.items():
			if len(files) == 1:
				continue

			manual = [f for f in files if '-' in f and 'orig' not in f]
			keep = min(manual) if manual else f'{title}.en.srt'

			for f in files:
				if f != keep:
					os.remove(os.path.join(subs_path, f))
					print(f'Removed duplicate subtitle file: {f}')

	logger.close()

	audio_files = [f for f in os.listdir(subs_path) if f.endswith(('.m4a', '.mp4', '.opus', '.webm'))]

	if audio_files:
		warnings.simplefilter('ignore', category=UserWarning)
		warnings.simplefilter('ignore', category=FutureWarning)
		warnings.simplefilter('ignore', category=ReproducibilityWarning)

		model = whisperx.load_model(whisper_model, device=whisper_device, compute_type=whisper_compute_type, language='en')
		model_a, metadata = whisperx.load_align_model(language_code='en', device=whisper_device)
		subtitle_writer = get_writer('srt', subs_path)

		processed_dir = os.path.join(subs_path, 'processed')
		os.makedirs(processed_dir, exist_ok=True)

		starting_whisper_batch_size = whisper_batch_size
		for filename in audio_files:
			whisper_batch_size = starting_whisper_batch_size

			audio_path = os.path.join(subs_path, filename)
			audio = whisperx.load_audio(audio_path)

			aligned_result = None
			result = None

			while result == None:
				try:
					print(f'Generating subtitles for {filename}')
					result = model.transcribe(audio, batch_size=whisper_batch_size, chunk_size=10, combined_progress=False, print_progress=True, task='translate', verbose=False)
				except RuntimeError as e:
					print(f'Runtime Error: {e}')
					if e.args[0] == 'CUDA failed with error out of memory':
						whisper_batch_size = whisper_batch_size - 1
						if whisper_batch_size < 1:
							whisper_batch_size = 1
						print(f'Lowering batch size to: {whisper_batch_size}')
				except Exception as e:
					print(f'Error: {e}')
					print('Retrying...')

			print(f'Aligning subtitles for {filename}')
			aligned_result = whisperx.align(result['segments'], model_a, metadata, audio, whisper_device, return_char_alignments=False)
			aligned_result['language'] = result['language']
			subtitle_writer(aligned_result, f'{os.path.splitext(filename)[0]}.srt', {'max_line_width': None, 'max_line_count': None, 'highlight_words': False})
			shutil.move(audio_path, os.path.join(processed_dir, filename))

if __name__ == '__main__':
	parser = argparse.ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, description='Rodney guy, rodney guy.')
	parser.add_argument('--urls', type=str, nargs='+', default=[
		'https://youtube.com/playlist?list=PL0WntInWBYyDH6wiNKavhF7HI5nERWDKy'
	], help='List of URLs to obtain subtitles for.')
	parser.add_argument('--cookies-path', type=str, default='cookies.txt', help='Netscape formatted file to read cookies from, can be obtained with: yt-dlp --cookies-from-browser firefox --cookies cookies.txt')
	parser.add_argument('--subs-path', type=str, default='./subtitles', help='Path to save files to.')
	parser.add_argument('--whisper-batch-size', type=int, default=8, help='Batch size to use for WhisperX transcription. Lower this if you run out of VRAM.')
	parser.add_argument('--whisper-compute-type', type=str, default='float16', help='Compute type to use for WhisperX transcription (int8/float16/float32).')
	parser.add_argument('--whisper-device', type=str, default='cuda', help='Whether to perform transcription on GPU (CUDA) or CPU.')
	parser.add_argument('--whisper-model', type=str, default='large-v3', help='Whisper model to use for transcription (see: https://github.com/openai/whisper#available-models-and-languages)')
	args = parser.parse_args()

	get_subtitles(args.urls, args.cookies_path, args.subs_path, args.whisper_model, args.whisper_compute_type, args.whisper_batch_size, args.whisper_device)
