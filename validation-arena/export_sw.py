#!/usr/bin/env python3
import tensorflow as tf
from huggingface_hub import snapshot_download
import tf2onnx


def main():
	print("Downloading model...")
	model_path = snapshot_download(repo_id='SmilingWolf/wd-v1-4-vit-tagger-v2', revision='v2.0')
	print(model_path)

	print("Loading model...")
	model = tf.keras.models.load_model(model_path)

	print("Converting model...")
	tf2onnx.convert.from_keras(model, output_path='wd-v1-4-vit-tagger-v2.onnx')


if __name__ == '__main__':
	main()
