#!/usr/bin/env python3
"""
Goes through all the images in the tag machine and inserts them into our auxiliary database.
"""
from tag_machine_api import DatabaseData
import logging
from utils import open_aux_db, batcher
from tqdm import tqdm


def main():
	# Set up logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Open auxiliary database
	db = open_aux_db()
	cursor = db.cursor()

	# Read tag data
	logging.info("Reading database data")
	db_data = DatabaseData()
	db_data.fetch_tags()
	db_data.fetch_images(with_blame=False)

	# Process
	hashes = ((image.id, bytes.fromhex(image.hash)) for image in db_data.images.values() if image.active)
	batches = batcher(hashes, 1000)
	n_batches = len(db_data.images) // 1000 + 1

	for batch in tqdm(batches, total=n_batches, desc="Inserting images", dynamic_ncols=True):
		# Insert into database
		cursor.executemany("INSERT INTO images (id, hash) VALUES (?, ?) ON CONFLICT DO NOTHING", batch)
		db.commit()


if __name__ == '__main__':
	main()