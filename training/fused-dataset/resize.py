#!/usr/bin/env python3
import os
import tempfile
from PIL import Image, features
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import argparse
import random
from datasets import load_dataset, DatasetDict


parser = argparse.ArgumentParser(description='Resize images')
parser.add_argument('--size', type=int, default=448, help='Size to resize to')
parser.add_argument('--workers', type=int, default=1, help='Number of workers to use')
parser.add_argument('--quality', type=str, action='append', help='Specify a WebP quality, method, and probability to use. Examples: 80,4,0.5. Can be specified multiple times. Probability must add up to 1. Use quality -1 for lossless.')
parser.add_argument('--cache-name', type=str, default='cache', help="Name of the cache folder")
parser.add_argument('--output-dir', type=str, default='cache', help='Output directory; defaults to cache; will be appended with cache name')
parser.add_argument('--dataset', type=str, required=True, help='Dataset to reference. Only images referenced by this dataset will be resized.')
parser.add_argument('--source', type=str, required=True, help='Source directory of images')


worker_output_dir: Path | None = None
worker_qualities: list[tuple[int, int, float]] | None = None
worker_size: int | None = None
worker_cache_name: str | None = None
worker_source_dir: Path | None = None


def main():
	args = parser.parse_args()

	args.output_dir = Path(args.output_dir)
	args.source = Path(args.source)

	# Parse qualities
	qualities = []

	if args.quality is not None:
		for quality in args.quality:
			parts = quality.split(',')
			if len(parts) != 3:
				raise ValueError(f'Invalid quality: {quality}')
			qualities.append((int(parts[0]), int(parts[1]), float(parts[2])))
			assert qualities[-1][0] >= -1 and qualities[-1][0] <= 100 and qualities[-1][1] >= 0 and qualities[-1][1] <= 6
	
	if len(qualities) == 0:
		raise ValueError('Must specify at least one quality')
	
	if sum(q[2] for q in qualities) != 1:
		raise ValueError('Probabilities must add up to 1')
	
	# Dataset
	dataset = load_dataset(args.dataset)
	assert isinstance(dataset, DatasetDict)
	dataset_hashes = set(str(h.hex()) for h in dataset['train']['hash'])
	dataset_hashes.update(str(h.hex()) for h in dataset['validation']['hash'])

	# Check for libjpeg-turbo
	if features.check_feature('libjpeg_turbo'):
		print('Using libjpeg-turbo')
	else:
		print('Not using libjpeg-turbo')

	# Check for Pillow-SIMD
	if 'post' in Image.__version__:
		print('Using Pillow-SIMD')
	else:
		print('Not using Pillow-SIMD')
	
	# Resize
	with multiprocessing.Pool(args.workers, initializer=init_worker, initargs=(qualities, args.size, args.output_dir, args.cache_name, args.source)) as pool:
		for _ in tqdm(pool.imap_unordered(resize_worker, dataset_hashes), total=len(dataset_hashes), smoothing=0.05, dynamic_ncols=True):
			pass


def load_and_crop_image(path, size) -> Image.Image:
	img = Image.open(path)
	
	# Convert to RGB
	img = img.convert('RGB')
	
	# Resize the image so the longest side is size pixels long
	if img.size[0] > img.size[1]:
		new_width = size
		new_height = int(img.size[1] * size / img.size[0])
	else:
		new_width = int(img.size[0] * size / img.size[1])
		new_height = size
	
	if new_width != img.size[0] or new_height != img.size[1]:
		img = img.resize((new_width, new_height), Image.BICUBIC)
	
	# Always paste, to nuke color profiles from the source image, which can cause issues
	new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
	new_image.paste(img, (0, 0))

	return new_image


def init_worker(qualities: list[tuple[int, int, float]], size: int, dst_folder: Path, cache_name: str, src_folder: Path):
	global worker_qualities, worker_size, worker_output_dir, worker_cache_name, worker_source_dir
	worker_qualities = qualities
	worker_size = size
	worker_output_dir = Path(dst_folder)
	worker_cache_name = cache_name
	worker_source_dir = Path(src_folder)


def resize_worker(hash: str):
	global worker_qualities, worker_size, worker_output_dir, worker_cache_name, worker_source_dir
	assert worker_qualities is not None and worker_size is not None and worker_output_dir is not None and worker_cache_name is not None and worker_source_dir is not None

	resized_path = worker_output_dir / worker_cache_name / f"{hash[:2]}" / f"{hash[2:4]}" / f"{hash}.webp"

	if resized_path.exists():
		return
	
	original_path = worker_source_dir / hash[:2] / hash[2:4] / hash
	
	img = load_and_crop_image(original_path, worker_size)
	quality = random.choices([q[:2] for q in worker_qualities], [q[2] for q in worker_qualities])[0]
	
	resized_path.parent.mkdir(parents=True, exist_ok=True)
	with tempfile.NamedTemporaryFile(delete=False, dir=worker_output_dir) as tmp:
		if quality[0] == -1:
			img.save(tmp.name, 'webp', lossless=True)
		else:
			img.save(tmp.name, 'webp', quality=quality[0], method=quality[1])
		os.rename(tmp.name, resized_path)


if __name__ == '__main__':
	main()