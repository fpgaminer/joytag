#!/usr/bin/env python3
import math
from pathlib import Path
from utils import batcher, open_aux_db
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from danbooru_metadata import TagMappings
import logging
from tag_machine_api import DatabaseData


VALIDATION_SIZE = 2**15
MIN_TAG_COUNT = 1200  # Minimum number of times a tag must be used to be included in the dataset


# Schema for the metadata file
schema = pa.schema([
	pa.field("post_id", pa.int64()),
	pa.field("tags", pa.list_(pa.int16())),
	pa.field("hash", pa.binary()),  # sha256 hash of the source file
])


@dataclass
class Record:
	post_id: int
	tags: set[str]
	hash: bytes


@dataclass
class RecordInt:
	post_id: int
	tags: set[int]
	hash: bytes


def main():
	# Set up logging
	logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

	# Aux DB
	logging.info("Reading aux database")
	aux_db = open_aux_db()
	cursor = aux_db.cursor()
	cursor.execute("SELECT hash, trainable_image, phash FROM images")
	hash_to_aux: dict[bytes, tuple[bool, int | None]] = {bytes(row[0]): (row[1] != 0, row[2]) for row in cursor}

	# Read tag data
	logging.info("Reading database data")
	tag_mappings = TagMappings('../metadata')
	db_data = DatabaseData()
	db_data.fetch_tags()
	#db_data.fetch_images(with_blame=True)

	# Read all images in the tag database
	# A "post" consists of an image id mapped to a set of tags
	# We also collect source counts and tag counts
	# phash_to_id and filehash_to_id are used for deduplication
	# NOTE: Tag machine should already enforce that images are unique by hash, but we do it again here just in case
	logging.info("Collecting posts")
	posts: dict[int, Record] = {}
	source_counts = defaultdict(int)
	phash_to_id: dict[int, int] = {}
	filehash_to_id: dict[bytes, int] = {}

	#for image in db_data.images.values():
	for image in tqdm(db_data.fetch_image_batches(with_blame=True), desc="Fetching images", dynamic_ncols=True, total=len(hash_to_aux)):
		if not image.active:
			continue

		assert image.tags_blame is not None

		trainable, phash = hash_to_aux[bytes.fromhex(image.hash)]

		if not trainable:
			continue

		# Include all danbooru images
		# For all other images, include only images with 10 or more tags by user 1
		has_human_tags = sum(1 for user_id in image.tags_blame.values() if user_id == 1) >= 10

		if 'danbooru_post_id' in image.attributes:
			pass
		elif not has_human_tags:
			continue

		# Build the set of tags
		tags = set(tag_mappings.get_canonical(tag) for tag in image.tags)
		for tag in tags:
			tags.update(tag_mappings.get_implications(tag))
		
		# Remove blacklisted tags and deprecations
		tags.difference_update(tag_mappings.blacklist)
		tags.difference_update(tag_mappings.deprecations)

		# Deduplication
		# If a duplicate is encountered, we only keep the first one and merge the tags
		if image.hash in filehash_to_id:
			post_id = filehash_to_id[image.hash]
			post = posts[post_id]
			post.tags.update(tags)
			continue
		elif phash is not None and phash in phash_to_id:
			post_id = phash_to_id[phash]
			post = posts[post_id]
			post.tags.update(tags)
			continue

		posts[image.id] = Record(
			post_id=image.id,
			tags=tags,
			hash=bytes.fromhex(image.hash),
		)
		if phash is not None:
			phash_to_id[phash] = image.id
		filehash_to_id[bytes.fromhex(image.hash)] = image.id

		if 'source' in image.attributes:
			source_counts[image.attributes['source']] += 1
	
	logging.info(f"Found {len(posts)} posts")
	logging.info(f"Sources: {source_counts}")

	# Remove posts with less than 5 tags, as those are probably not properly tagged.
	posts = {post_id: post for post_id, post in posts.items() if len(post.tags) >= 5}
	logging.info(f"Found {len(posts)} posts with at least 5 tags")
	
	# Count tag usage
	tag_counts: dict[str, int] = defaultdict(int)
		
	for post in posts.values():
		for tag in post.tags:
			tag_counts[tag] += 1

	# Top tags
	logging.info(f"Found {sum(1 for tag, count in tag_counts.items() if count >= 10000)} tags with at least 10,000 usage")
	logging.info(f"Found {sum(1 for tag, count in tag_counts.items() if count >= 2000)} tags with at least 2,000 usage")
	logging.info(f"Found {sum(1 for tag, count in tag_counts.items() if count >= 1200)} tags with at least 1,200 usage")
	logging.info(f"Found {sum(1 for tag, count in tag_counts.items() if count >= 1000)} tags with at least 1,000 usage")
	tag_counts = {tag: count for tag, count in tag_counts.items() if count >= MIN_TAG_COUNT}
	top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
	top_tags = [tag for tag, _ in top_tags]
	tag_to_id = {tag: i for i, tag in enumerate(top_tags)}
	logging.info(f"Found {len(top_tags)} tags with at least {MIN_TAG_COUNT} usage")

	with open('top_tags.txt', 'w') as f:
		f.write('\n'.join(top_tags))
	
	# Convert to RecordInt
	records = {post_id: RecordInt(post_id=post_id, tags={tag_to_id[tag] for tag in post.tags if tag in tag_to_id}, hash=post.hash) for post_id, post in posts.items()}

	# Build validation set
	validation_ids = build_validation_set(records)
	
	# Remove validation set from training set
	training_ids = set(records.keys()) - validation_ids

	# Write validation
	logging.info("Writing validation set")
	dataset_writer('validation.parquet', [records[post_id] for post_id in validation_ids])

	# Write training
	logging.info("Writing training set")
	dataset_writer('train.parquet', [records[post_id] for post_id in training_ids])


def build_validation_set(records: dict[int, RecordInt]) -> set[int]:
	"""
	We loop until the validation set is representative of the full dataset.
	"""
	max_tag_id = max(max(post.tags) for post in records.values())
	all_ids = list(records.keys())
	np.random.default_rng(42).shuffle(all_ids)

	validation_ids: set[int] = set()
	validation_tag_counts = {tag_id: 0 for tag_id in range(max_tag_id + 1)}
	representative = False

	for post_id in all_ids:
		if len(validation_ids) >= VALIDATION_SIZE:
			break

		post = records[post_id]

		# Initially we want to ensure that each post added to the validation set is increasing the tag counts
		# for any tag with less than 5 occurrences thus far.
		# After that, random sampling is fine.
		if not representative:
			helpful_tags = [tag for tag in post.tags if validation_tag_counts[tag] < 5]

			if len(helpful_tags) == 0:
				continue

		validation_ids.add(post_id)

		for tag in post.tags:
			validation_tag_counts[tag] += 1
		
		if not representative:
			representative = all(count >= 5 for count in validation_tag_counts.values())

	assert representative, "Validation set is not representative of the full dataset"
	assert len(validation_ids) == VALIDATION_SIZE, f"Validation set is not the correct size: {len(validation_ids)}"
	return validation_ids


def dataset_writer(dest_path: Path | str, records: list[RecordInt]):
	with pq.ParquetWriter(dest_path, schema) as writer:
		for batch in tqdm(batcher(records, 1000), total=math.ceil(len(records) / 1000), dynamic_ncols=True):
			post_ids = [info.post_id for info in batch]
			tags = [info.tags for info in batch]
			hashes = [bytes(info.hash) for info in batch]

			batch = pa.RecordBatch.from_arrays([
				pa.array(post_ids, type=pa.int64()),
				pa.array(tags, type=pa.list_(pa.int16())),
				pa.array(hashes, type=pa.binary()),
			], schema=schema)
			writer.write(batch)


if __name__ == '__main__':
	main()