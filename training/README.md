# Training

Scripts used during the training of JoyTag.

WARNING: WIP


* fused-dataset
	* See `fused-dataset/README.md` for detailed information.
	* This set of scripts builds the training and validation datasets.
	* The end result is a dataset uploaded to HuggingFace with the schema: post_id (int64), tags (list of int16), hash (binary)
	* The training scripts work off this dataset in addition to the resized images stored on the filesystem.