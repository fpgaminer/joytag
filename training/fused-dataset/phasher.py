#!/usr/bin/env python3
import subprocess
from utils import open_aux_db, TAG_MACHINE_PATH, batcher
import logging
from pathlib import Path
from tqdm import tqdm
from tempfile import NamedTemporaryFile
import ctypes


HASHER_PATH = '/home/night/rust-phash-hasher/target/release/hasher'


def main():
	# Set up logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Open auxiliary database
	db = open_aux_db()
	cursor = db.cursor()

	# Process
	cursor.execute("SELECT id, hash FROM images WHERE phash IS NULL")
	work = [(int(row[0]), bytes(row[1])) for row in cursor]
	n_images = len(work)

	batches = batcher(work, 1000)
	with tqdm(total=n_images, desc="Processing images", dynamic_ncols=True) as pbar, NamedTemporaryFile() as f:
		temp_path = Path(f.name)

		for batch in batches:
			file_hash_to_id = {job[1].hex(): job[0] for job in batch}
			pathes = [str(job_to_path(job)) for job in batch]
			input_data = "\n".join(pathes).encode('utf-8')

			# Run hasher
			f.truncate(0)
			outputs = subprocess.check_output([HASHER_PATH, "--output", str(temp_path)], input=input_data)
			f.seek(0)
			outputs = f.read().decode('utf-8').split('\n')

			# Split and parse outputs
			outputs = (line.strip().split() for line in outputs if line.strip() != '')
			outputs = ((Path(line[0]).stem, int(line[1])) for line in outputs)

			# Convert the paths back to image IDs, and convert the phashes to signed 64-bit integers (for sqlite)
			outputs = ((ctypes.c_int64(phash).value, file_hash_to_id[file_hash]) for file_hash, phash in outputs)

			# Update database
			cursor.executemany("UPDATE images SET phash=? WHERE id=?", outputs)
			db.commit()

			pbar.update(len(batch))


def job_to_path(job: tuple[int, bytes]) -> Path:
	file_hash = job[1].hex()
	return TAG_MACHINE_PATH / file_hash[:2] / file_hash[2:4] / file_hash


if __name__ == '__main__':
	main()