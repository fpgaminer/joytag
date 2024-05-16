# Fused Dataset

Fusing all datasets from the tag-machine, into a single dataset for the training scripts.

Run in roughly this order:

* `import_images.py`
	* Goes through all the images in the tag machine and inserts them into aux.sqlite3
* `phasher.py`
	* Computes the phash of all the images in aux.sqlite3. This is used to find duplicates.
* `trainable.py`
	* Goes through all the images in aux.sqlite3 and determines if they're "trainable" or not (non-animated, etc).
* `build.py`
	* Builds the final dataset, with all the images in aux.sqlite3 that are "trainable", not duplicates, have enough human tags, etc.
	* Produces `train.parquet`, `validation.parquet`, and `top_tags.txt`.
	* Parquet schema: post_id (int64), tags (list of int16), hash (binary)
	* post_id is a unique identifer for a "post" (image-text combination) in the tag machine.
	* hash is the sha256 hash of the original image file. The training script uses this to find the image file on the filesystem, e.g. `../cache/512-dataset-rng/hash[:2]/hash[2:4]/hash[4:]`.
* `upload.py`
	* Uploads the final dataset to HuggingFace.
	* Schema is the same. Tag names are added as metadata.
* `resize.py`
	* Resize images referenced by the specified dataset.
	* Example: `./resize.py --dataset fancyfeast/danbooru2021-slim-v3 --size 512 --workers 8 --output-dir ../cache --cache-name 512-dataset-rng --quality='90,6,.45' --quality='100,6,.45' --quality='-1,6,.1' --source ~/tag-machine/rust-api/images`
	* The example produces resized images compressed with webp, with 3 different quality settings: 45% of the images will be at 90% lossy quality; 45% at 100% lossy quality; and 10% at lossless quality.
	* This helps reduce the size of the training dataset, while hopefully not corrupting the data with compression too much.