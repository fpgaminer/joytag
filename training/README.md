# Training

Scripts used during the training of JoyTag.


* `fused-dataset/`
	* See `fused-dataset/README.md` for detailed information.
	* This set of scripts builds the training and validation datasets.
	* The end result is a dataset uploaded to HuggingFace with the schema: post_id (int64), tags (list of int16), hash (binary)
	* The training scripts work off this dataset in addition to the resized images stored on the filesystem.

* `train/`
	* `Train.py` is the main training script.
	* Example: `torchrun --standalone --nproc_per_node=8 Train.py --dataset-path fancyfeast/danbooru2021-slim-v3 --images-path data/512-dataset-rng --device-batch-size 512 --wandb-project danbooru2021-embeddings-v15 --data-augment trivial2`