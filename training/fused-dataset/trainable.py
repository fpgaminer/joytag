#!/usr/bin/env python3
"""
Goes through all the images in the aux database and checks if they are "trainable" (i.e. not a gif, not transparent, etc.).
"""
import logging
from utils import open_aux_db, batcher, TAG_MACHINE_PATH
from PIL import Image
from tqdm import tqdm
import multiprocessing


def main():
	# Set up logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Open auxiliary database
	db = open_aux_db()
	cursor = db.cursor()

	# Process
	cursor.execute("SELECT id, hash FROM images WHERE trainable_image IS NULL")
	work = [(int(row[0]), bytes(row[1])) for row in cursor]
	n_images = len(work)

	with multiprocessing.Pool(8) as pool, tqdm(total=n_images, desc="Processing images", dynamic_ncols=True) as pbar:
		results = pool.imap_unordered(process_image, work)
		for batch in batcher(results, 1000):
			cursor.executemany("UPDATE images SET trainable_image=? WHERE id=?", batch)
			db.commit()
			pbar.update(len(batch))


def process_image(job: tuple[int, bytes]) -> tuple[bool, int]:
	image_id, file_hash = job
	return (is_image_trainable(file_hash), image_id)


def has_transparency(img):
	if img.info.get('transparency', None) is not None:
		return True
	if img.mode == 'P':
		transparent = img.info.get('transparency', -1)
		for _, index in img.getcolors():
			if index == transparent:
				return True
	elif img.mode == 'RGBA':
		extrema = img.getextrema()
		if extrema[3][0] < 255:
			return True
	
	return False


def is_image_trainable(file_hash: bytes) -> bool:
	"""
	Checks if the image is "trainable" (i.e. not a gif, not transparent, etc.).
	"""
	file_hash_hex = file_hash.hex()
	path = TAG_MACHINE_PATH / file_hash_hex[:2] / file_hash_hex[2:4] / file_hash_hex

	try:
		img = Image.open(path)
		img.load()
	except:
		return False
	
	# Check if gif
	if img.format == 'GIF' or (hasattr(img, 'is_animated') and img.is_animated):
		return False
	
	# Check if has transparency
	if has_transparency(img):
		return False
	
	# Convert to RGB
	# NOTE: An image caused a crash here, so wrapping in a try/except
	try:
		img = img.convert('RGB')
	except:
		return False
	
	return True


if __name__ == '__main__':
	main()