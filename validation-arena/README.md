# Validation Arena

This code calculates metrics for different tagging models using unseen images, so that performance can be compared.  Comparing each model's stated metrics is usually not a good way to compare performance, due to differences in validation sets, methodology, etc.  So instead, a set of recent Danbooru images are downloaded and used as a common set of unseen images.  Since the images are recent, they are unlikely to have been in any training set.  They are also likely to be a slight domain shift, testing the ability of the models to generalize.

It is recommended to use images that are at least a couple of months old on Danbooru, so that they are more likely to be properly tagged.  As of now, a starting ID of 6800000 is hardcoded into the scripts.

## Results

| Model       | Precision | Recall  | F1     | AP     |
|-------------|-----------|---------|--------|--------|
| SmilingWolf | 0.5156    | 0.3943  | 0.4241 | 0.4368 |
| JoyTag      | 0.4377    | 0.4264  | 0.4179 | 0.4040 |

32768 posts, min id 6734042, max id 6769098
3993 tags in common after filtering by usage
SmilingWolf model: wd-v1-4-vit-tagger-v2

## Usage

NOTE: Uses danbooru_metadata library (https://github.com/fpgaminer/danbooru-metadata) and metadata from danbooru2021 datasest.

* `download.py`
	* Downloads posts and images from Danbooru, starting at a given ID and descending.
	* Images are saved to `originals/` by post ID.
	* Posts are saved to `posts.jsonl`.

* `trainable.py`
	* Validates which images can be loaded, have no transparency, aren't animated, etc.
	* Valid images are resized to be below or within 1024x1024 and saved to `resized` using WebP 80%.
	* This compression has a negligible effect on predictions, but saves a lot of disk space.

* `export_sw.py`
	* Exports the SmilingWolf model to ONNX format.
	* This is needed because the current ONNX model on HuggingFace has a fixed batch size of 1.
	* The model is exported to the local directory as `wd-v1-4-vit-tagger-v2.onnx`.

* `validate.py`
	* Runs inference using both models and reports global mean Precision, Recall, F1, and Average Precision.
	* Only tags used by both models are considered.
	* Only tags which occur sufficiently in the dataset are considered.
	* Predictions are saved to `*_predictions.pt` for later analysis, along with `ground_truth.py`.
	* The post IDs used are saved to `validation_post_ids.txt` for reproducibility.
	* Detailed per-tag metrics are saved to `detailed_metrics.txt`.
	* All metrics are calculated with respect to tags.  i.e. the reported F1 score is first calculated for each tag, and then meaned across tags.
