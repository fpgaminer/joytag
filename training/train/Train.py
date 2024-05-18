#!/usr/bin/env python3
import json
import logging
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import datasets
import numpy as np
import omegaconf
import torch
import torch._dynamo
import torch._dynamo.config
import torch.amp
import torch.backends.cuda
import torch.backends.cudnn
import torch.distributed
import torch.utils.data
from omegaconf import MISSING
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.optim.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import wandb
from data import BetterDistributedSampler, GPUDataProcessing, ImageDataset
from lamb import Lamb
from Models import MODEL_CONFIGS, VisionModel
from utils import MetricCounters, distributed_cleanup, distributed_rank, distributed_setup, distributed_world_size, log_rank_0, parse_args_into_config, temprngstate, get_cosine_schedule_with_warmup


@dataclass
class Config:
	# General settings
	output_dir: Path = Path("checkpoints")               # Output directory
	dataset_path: str = MISSING                          # Path to dataset to use
	dataset_revision: Optional[str] = None               # Dataset revision
	images_path: Path = MISSING                          # Path to images
	resume: Optional[Path] = None                        # Resume from checkpoint
	device_batch_size: int = 1                           # Device batch size
	wandb_project: Optional[str] = None                  # Wandb project
	save_every: int = 1000000                            # Save a checkpoint every n samples (approx)
	save_everything: bool = True                         # Save everything
	save_only_latest: bool = True                        # Only save the latest checkpoint, overwriting the previous one as we go
	test_every: int = 1000000                            # Test every n samples (approx)
	seed: int = 42                                       # Random seed
	allow_tf32: bool = True                              # Allow tf32
	model: str = "SWModel20"                             # Model config to use
	num_workers: int = 8                                 # Number of workers
	minimum_training_tags: int = 5                       # Skip images with fewer than this many tags
	image_size: int = 224                                # Image size
	loss_type: str = "focal2"							 # Loss type
	finetune_path: Optional[Path] = None                 # Path to finetune from

	# Training settings
	batch_size: int = 4096                               # Actual batch size; gradient accumulation is used on device_batch_size to achieve this
	learning_rate: float = 4e-3                          # Learning rate
	min_lr_ratio: float = 0.0                            # Minimum learning rate ratio for scheduler
	warmup_samples: int = 10000                          # Warmup samples
	max_samples: int = 220000000                         # Max samples trained for in this session
	lr_scheduler_type: str = "cosine"                    # Learning rate scheduler type
	data_augment: Optional[str] = None                   # Data augmentation type (One of: 'trivial', 'sw', 'flip', None)
	mixup: float = 0.0                                   # Mixup alpha
	clip_grad_norm: Optional[float] = 1.0                # Clip gradient norm
	label_smoothing: float = 0.0                         # Label smoothing

	# Optimizer
	optimizer_type: str = "lamb"                         # Optimizer type
	adam_beta1: float = 0.9                              # Adam beta1
	adam_beta2: float = 0.999                            # Adam beta2
	adam_eps: float = 1e-6                               # Adam epsilon
	adam_weight_decay: float = 0.05                      # Adam weight decay

	# Progressive resizing
	progressive_resizing: bool = False                   # Progressive resizing
	progressive_resizing_initial_scale: float = 0.5      # Progressive resizing initial scale
	progressive_resizing_warmup: float = 0.5             # How much of the training cycle will use initial scale
	progressive_resizing_finetune: float = 0.2           # How much of the training cycle will use full size
	progressive_resizing_increments: int = 4             # Ensures progressive image size is divisible by this number

	# Progressive patch dropout
	progressive_patch_dropout: bool = False              # Progressive patch dropout
	progressive_patch_dropout_initial_rate: float = 0.5  # Progressive patch dropout initial rate
	progressive_patch_dropout_warmup: float = 0.5        # How much of the training cycle will use initial rate
	progressive_patch_dropout_finetune: float = 0.2      # How much of the training cycle will use full rate

	# Misc
	use_amp: bool = True                                 # Use AMP fp16
	use_zero_optimizer: bool = False                     # Set to true to use the ZeroRedundancyOptimizer when training with multiple GPUs
	model_dtype: str = "float32"                         # Model dtype
	compile_model: bool = True                           # Compile model

	# Gradient scaler
	grad_scaler_enabled: bool = True                     # Use GradScaler
	grad_scaler_init: float = 2**16                      # Initial scale
	grad_scaler_interval: int = 4000000                  # Update interval (in samples) N.B. Do not change after training has started; PyTorch does not support this due to using `step == interval` instead of `step >= interval`


class MainTrainer:
	config: Config
	rank: int
	logger: logging.Logger
	device: str
	world_size: int
	run_id: str
	device_batch_size: int
	gradient_accumulation_steps: int
	test_every: int
	save_every: int
	model_config: dict[str, Any]
	model: VisionModel
	compiled_model: torch.nn.Module | torch.nn.parallel.DistributedDataParallel
	train_dataloader: torch.utils.data.DataLoader
	validation_dataloader: torch.utils.data.DataLoader | None
	num_tags: int
	source_dataset: datasets.DatasetDict

	def __init__(self, config: Config, run_id: str, logger: logging.Logger):
		self.config = config
		self.rank = distributed_rank()
		self.logger = logger
		self.device = f'cuda:{self.rank}'
		self.world_size = distributed_world_size()
		self.run_id = run_id
		self.old_checkpoint = None

		self.config.resume = Path(self.config.resume) if self.config.resume is not None else None
		self.config.finetune_path = Path(self.config.finetune_path) if self.config.finetune_path is not None else None
		self.config.output_dir = Path(self.config.output_dir)
		self.config.images_path = Path(self.config.images_path)
		
		if self.config.allow_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
		
		# Increase compilation cache for now
		#torch._dynamo.config.cache_size_limit = 128
		
		self.device_batch_size = min(self.config.batch_size // self.world_size, config.device_batch_size)
		self.gradient_accumulation_steps = self.config.batch_size // (self.device_batch_size * self.world_size)
		self.test_every = int(math.ceil(config.test_every / self.config.batch_size))
		self.save_every = int(math.ceil(config.save_every / self.config.batch_size))

		assert self.config.batch_size == self.device_batch_size * self.gradient_accumulation_steps * self.world_size, f"batch_size {self.config.batch_size} must be divisible by device_batch_size {config.device_batch_size}"
	
	def load_models(self):
		log_rank_0(self.logger, logging.INFO, "Building model...")

		if self.config.model not in MODEL_CONFIGS:
			raise ValueError(f"Unknown model type {self.config.model}")
		
		self.model_config = MODEL_CONFIGS[self.config.model]
		self.model_config['image_size'] = self.config.image_size
		self.model_config['n_tags'] = self.num_tags
		self.model_config['loss_type'] = self.config.loss_type
		self.model_config['use_amp'] = self.config.use_amp

		self.model = VisionModel.from_config(self.model_config)
		log_rank_0(self.logger, logging.INFO, "Model built from config")

		dtype_dict = { "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16 }
		self.model = self.model.to(dtype=dtype_dict[self.config.model_dtype])

		if self.config.resume is not None:
			self.model.load(torch.load(self.config.resume / "model.pt", map_location='cpu'))
		elif self.config.finetune_path is not None:
			self.model.load(torch.load(self.config.finetune_path, map_location='cpu'))

		self.model.to(self.device)
		self.logger.info("Model moved to device")
		self.compiled_model = self.model

		# Handle distributed training
		if self.world_size > 1:
			self.compiled_model = torch.nn.parallel.DistributedDataParallel(self.compiled_model, device_ids=[self.rank], output_device=self.rank, gradient_as_bucket_view=True, find_unused_parameters=True)
			log_rank_0(self.logger, logging.INFO, "Model wrapped in DistributedDataParallel")
	
	def build_optimizer(self):
		log_rank_0(self.logger, logging.INFO, "Building optimizer...")

		assert self.model is not None
		params = list(self.model.get_optimized_parameters(self.config.learning_rate))

		if self.config.optimizer_type == 'adam':
			optimizer_cls = torch.optim.Adam
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'adamw':
			optimizer_cls = torch.optim.AdamW
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'lamb':
			optimizer_cls = Lamb
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		elif self.config.optimizer_type == 'fusedlamb':
			from apex.optimizers import FusedLAMB
			optimizer_cls = FusedLAMB
			kwargs = {
				'lr': self.config.learning_rate,
				'betas': (self.config.adam_beta1, self.config.adam_beta2),
				'eps': self.config.adam_eps,
				'weight_decay': self.config.adam_weight_decay,
			}
		else:
			raise ValueError(f"Unknown optimizer type {self.config.optimizer_type}")
		
		if self.world_size > 1 and self.config.use_zero_optimizer:
			self.optimizer = ZeroRedundancyOptimizer(params, optimizer_class=optimizer_cls, parameters_as_bucket_view=True, **kwargs)
		else:
			self.optimizer = optimizer_cls(params, **kwargs)
		
	def build_datasets(self):
		log_rank_0(self.logger, logging.INFO, "Loading dataset...")

		# Load the dataset
		source_dataset = datasets.load_dataset(self.config.dataset_path, num_proc=8, revision=self.config.dataset_revision)
		assert isinstance(source_dataset, datasets.DatasetDict), "Dataset must be a DatasetDict"
		self.source_dataset = source_dataset
		
		source_dataset.set_format(type='torch', columns=['post_id', 'tags'], output_all_columns=True)

		# Filter out posts with too few tags
		min_training_tags = self.config.minimum_training_tags   # Save this variable locally so huggingface datasets can properly cache this filter
		train_dataset = source_dataset["train"].filter(lambda x: len(x['tags']) >= min_training_tags, num_proc=8)

		# Number of tags
		self.num_tags = len(source_dataset['train'].features['tags'].feature.names)
		log_rank_0(self.logger, logging.INFO, f"Number of tags: {self.num_tags}")

		# Build datasets
		self.train_dataset = ImageDataset(train_dataset, self.num_tags, self.config.images_path, data_augmentation=self.config.data_augment, label_smoothing=self.config.label_smoothing)
		self.validation_dataset = ImageDataset(source_dataset["validation"], self.num_tags, self.config.images_path, data_augmentation=None, label_smoothing=self.config.label_smoothing)
	
	def build_dataloader(self):
		log_rank_0(self.logger, logging.INFO, "Building dataloader...")

		self.train_sampler = BetterDistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True, drop_last=True, seed=self.config.seed, target_image_size=self.config.image_size)
		self.validation_sampler = BetterDistributedSampler(self.validation_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False, drop_last=False, seed=self.config.seed, target_image_size=self.config.image_size)

		self.train_dataloader = torch.utils.data.DataLoader(
			self.train_dataset,
			batch_size=self.device_batch_size,
			sampler=self.train_sampler,
			num_workers=self.config.num_workers,
			pin_memory=True,
			drop_last=True,
			pin_memory_device=self.device,
			persistent_workers=True,
		)
		
		self.validation_dataloader = torch.utils.data.DataLoader(
			self.validation_dataset,
			batch_size=self.device_batch_size,
			sampler=self.validation_sampler,
			num_workers=self.config.num_workers // 2,
			pin_memory=True,
			drop_last=False,
			pin_memory_device=self.device,
			persistent_workers=True,
		)
	
	def build_lr_scheduler(self):
		num_warmup_steps = int(math.ceil(self.config.warmup_samples / self.config.batch_size))

		if self.config.lr_scheduler_type == 'cosine':
			self.lr_scheduler = get_cosine_schedule_with_warmup(
				optimizer=self.optimizer,
				num_warmup_steps=num_warmup_steps,
				num_training_steps=self.total_steps,
				min_lr_ratio=self.config.min_lr_ratio,
			)
		elif self.config.lr_scheduler_type == 'onecycle':
			self.lr_scheduler = OneCycleLR(
				optimizer=self.optimizer,
				max_lr=self.config.learning_rate,
				total_steps=self.total_steps,
			)
		elif self.config.lr_scheduler_type == 'invsqrt':
			num_warmup_steps = max(1, num_warmup_steps)
			peak = num_warmup_steps ** 0.5

			def lr_lambda(step):
				if step < num_warmup_steps:
					return float(step) / float(num_warmup_steps)
				return peak * (step ** -0.5)
			
			self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
		else:
			raise ValueError(f"Unknown lr_scheduler_type {self.config.lr_scheduler_type}")
	
	def train(self):
		# Seed (each rank gets a unique seed derived from the global seed)
		seed = hash((self.config.seed, self.rank)) & 0xffffffff  # NumPy requires 32 bit seeds
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		np.random.seed(seed)
		random.seed(seed)

		# Set everything up
		self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=self.config.grad_scaler_enabled, init_scale=self.config.grad_scaler_init, growth_interval=(self.config.grad_scaler_interval // self.config.batch_size))
		self.build_datasets()
		self.load_models()
		self.build_optimizer()
		self.build_dataloader()
		self.total_steps = self.config.max_samples // self.config.batch_size
		self.build_lr_scheduler()
		gpu_data_processing = GPUDataProcessing(mixup_alpha=self.config.mixup)

		device_step = 0
		total_device_batches = self.total_steps * self.gradient_accumulation_steps

		assert self.compiled_model is not None and self.model is not None and self.train_dataloader is not None

		# Resume training states
		if self.config.resume is not None:
			resume = torch.load(self.config.resume / "training_state.pt", map_location='cpu')
			resume.update(torch.load(self.config.resume / f"training_state{self.rank}.pt", map_location='cpu'))

			random.setstate(resume['random_state'])
			np.random.set_state(resume['numpy_random_state'])
			torch.set_rng_state(resume['torch_random_state'])
			try:
				torch.cuda.set_rng_state(resume['torch_cuda_random_state'])
			except RuntimeError:
				self.logger.warning("Failed to restore cuda random state, this is normal if you're using a different number of GPUs than last time")
			
			self.lr_scheduler.load_state_dict(resume['lr_scheduler'])
			self.scaler.load_state_dict(resume['scaler'])
			self.optimizer.load_state_dict(resume['optimizer'])
			self.train_sampler.set_state(resume['sampler_epoch'], resume['sampler_index'])

			device_step = (resume['global_step'] + 1) * self.gradient_accumulation_steps
			del resume
		
		# Compile the model
		if self.config.compile_model:
			self.compiled_model = torch.compile(self.compiled_model) # type: ignore

		# Wandb
		if self.rank == 0 and self.config.wandb_project is not None:
			wandb.watch(self.compiled_model, log_freq=100)

		log_rank_0(self.logger, logging.INFO, "Starting training...")
		loss_sum = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)
		train_dataloader_iter = None

		pbar = tqdm(total=total_device_batches * self.world_size * self.device_batch_size, initial=device_step * self.world_size * self.device_batch_size, dynamic_ncols=True, disable=self.rank != 0, smoothing=0.01)
		with logging_redirect_tqdm():
			for device_step in range(device_step, total_device_batches):
				self.global_step = device_step // self.gradient_accumulation_steps
				self.global_samples_seen = (device_step + 1) * self.device_batch_size * self.world_size

				self.compiled_model.train()

				# Progressive resizing
				# NOTE: The way things are set up right now, the target image size won't take effect until the next epoch
				# This is because the dataloader workers won't pick up the updated dataset until the dataloader runs out and is re-iterated
				if self.config.progressive_resizing:
					global_step_ratio = self.global_step / self.total_steps

					if global_step_ratio < self.config.progressive_resizing_warmup:
						scale = self.config.progressive_resizing_initial_scale
					elif global_step_ratio < (1.0 - self.config.progressive_resizing_finetune):
						x = global_step_ratio - self.config.progressive_resizing_warmup
						x = x / (1.0 - self.config.progressive_resizing_warmup - self.config.progressive_resizing_finetune)
						scale = self.config.progressive_resizing_initial_scale + (1.0 - self.config.progressive_resizing_initial_scale) * x
					else:
						scale = 1.0
					
					target_size = int(self.config.image_size * scale)
					target_size = target_size - (target_size % self.config.progressive_resizing_increments)

					if target_size != self.train_sampler.target_image_size:
						# Clear the compilation cache, since the inputs to the model will be a different size
						torch._dynamo.reset()

					self.train_sampler.target_image_size = target_size
					self.validation_sampler.target_image_size = target_size
				
				# Progressive patch dropout
				if self.config.progressive_patch_dropout:
					global_step_ratio = self.global_step / self.total_steps

					x = global_step_ratio - self.config.progressive_patch_dropout_warmup
					x = x / (1.0 - self.config.progressive_patch_dropout_warmup - self.config.progressive_patch_dropout_finetune)
					x = max(0.0, x)
					rate = self.config.progressive_patch_dropout_initial_rate * (1.0 - x)
					rate = max(0.0, rate)

					self.model.patch_dropout = rate
				
				if train_dataloader_iter is None:
					train_dataloader_iter = iter(self.train_dataloader)
				
				# Reload the dataloader if necessary
				try:
					batch = next(train_dataloader_iter)
				except StopIteration:
					log_rank_0(self.logger, logging.INFO, "Reloading dataloader...")
					self.train_sampler.set_epoch(self.train_sampler.epoch + 1)  # This is important to ensure the data is re-shuffled after every use
					train_dataloader_iter = iter(self.train_dataloader)
					batch = next(train_dataloader_iter)
				
				# Move the batch to the device
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

				# GPU side data augmentation
				batch = gpu_data_processing(batch)

				# Before step
				before_step = getattr(self.model, 'before_step', None)
				if before_step is not None:
					before_step(steps=self.global_step, samples=self.global_samples_seen, optimizer=self.optimizer)

				is_last_device_step = (device_step + 1) % self.gradient_accumulation_steps == 0
				is_last_step = (self.global_step + 1) == self.total_steps

				# Forward pass
				# Disable gradient sync for all but the last device step
				with self.compiled_model.no_sync() if not is_last_device_step and self.world_size > 1 else nullcontext():
					preds = self.compiled_model(batch, return_loss=True)
					loss = preds['loss']
					loss = loss.float() / self.gradient_accumulation_steps
					
					loss_sum.add_(loss.detach())

					if torch.isnan(loss) or torch.isinf(loss):
						self.logger.error("ERROR: Loss is NaN or Inf")
						exit()
				
					# Backward pass
					self.scaler.scale(loss).backward()

				# Take a step if accumulation is complete
				if is_last_device_step:
					# Reduce loss_sum across devices for logging
					torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)

					# Unscale the gradients before clipping
					self.scaler.unscale_(self.optimizer)

					# Clip gradients
					if self.config.clip_grad_norm is not None:
						torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)

					# Take a step
					self.scaler.step(self.optimizer)
					self.scaler.update()
					self.lr_scheduler.step()
					#self.optimizer.zero_grad(set_to_none=True)
					self.optimizer.zero_grad()

					after_step = getattr(self.model, 'after_step', None)
					if after_step is not None:
						after_step(steps=self.global_step, samples=self.global_samples_seen, optimizer=self.optimizer)

					if self.rank == 0:
						logs = {
							"train/loss": loss_sum.item() / self.world_size,
							"train/lr": self.lr_scheduler.get_last_lr()[0],
							"train/samples": self.global_samples_seen,
							"train/scaler": self.scaler.get_scale(),
						}
						wandb.log(logs, step=self.global_step)
					
					loss_sum.zero_()
				
					# Save checkpoint
					# Saved every save_every steps and at the end of training
					if self.save_every > 0 and ((self.global_step + 1) % self.save_every == 0 or is_last_step):
						self.save_checkpoint()
						log_rank_0(self.logger, logging.INFO, f"Target Image Size: {self.train_sampler.target_image_size}")
					
					# Validation
					# Run every test_every steps and at the end of training
					if self.test_every > 0 and ((self.global_step + 1) % self.test_every == 0 or is_last_step):
						self.do_validation()
				
				pbar.update(self.world_size * self.device_batch_size)
		
		pbar.close()

	def save_checkpoint(self):
		assert self.model is not None and self.train_dataloader is not None
		log_rank_0(self.logger, logging.INFO, "Saving checkpoint...")

		if isinstance(self.optimizer, ZeroRedundancyOptimizer):
			log_rank_0(self.logger, logging.INFO, "Consolidating optimizer state...")
			self.optimizer.consolidate_state_dict(to=0)
			log_rank_0(self.logger, logging.INFO, "Done consolidating optimizer state")
		
		sampler_epoch = self.train_sampler.epoch
		sampler_index = self.global_samples_seen // self.world_size   # NOTE: sampler_index is in terms of "samples", not batches or steps
		sampler_index = sampler_index % (len(self.train_dataloader) * self.device_batch_size)

		base_path = self.config.output_dir / self.run_id
		path = base_path / f"samples_{self.global_samples_seen}"
		tmp_path = base_path / "tmp"
		tmp_path.mkdir(parents=True, exist_ok=True)

		if self.rank == 0:
			self.logger.info(f"Saving checkpoint to {path}...")

			# Model
			torch.save(self.model.save(), tmp_path / "model.pt")

			# Config
			with open(tmp_path / "config.json", 'w') as f:
				json.dump(self.model_config, f)
			
			# Labels
			with open(tmp_path / "top_tags.txt", 'w') as f:
				top_tags = self.source_dataset['train'].features['tags'].feature.names
				f.write("\n".join(top_tags))

			# Training state
			data = {
				"global_step": self.global_step,
				"global_samples_seen": self.global_samples_seen,
				"sampler_epoch": sampler_epoch,
				"sampler_index": sampler_index,
				"lr_scheduler": self.lr_scheduler.state_dict(),
			}

			if self.config.save_everything:
				data['optimizer'] = self.optimizer.state_dict()
			
			torch.save(data, tmp_path / "training_state.pt")

		# Rank dependent stuff
		if self.config.save_everything:
			torch.save({
				"global_step": self.global_step,
				"global_samples_seen": self.global_samples_seen,
				"scaler": self.scaler.state_dict(),   # NOTE: I'm fairly certain that GradScaler is rank independent, but I'm saving it here just in case
				"random_state": random.getstate(),
				"numpy_random_state": np.random.get_state(),
				"torch_random_state": torch.random.get_rng_state(),
				"torch_cuda_random_state": torch.cuda.random.get_rng_state(),
			}, tmp_path / f"training_state{self.rank}.pt")
		
		# Synchonize, so that all ranks are done before we move the checkpoint into place
		if self.world_size > 1:
			torch.distributed.barrier()
		
		if self.rank == 0:
			# Move the checkpoint into place
			tmp_path.rename(path)

			# Delete old checkpoint
			if self.config.save_only_latest and self.old_checkpoint is not None and self.old_checkpoint.exists():
				self.logger.info(f"Deleting old checkpoint {self.old_checkpoint}")
				for sub_path in self.old_checkpoint.rglob("*"):
					if sub_path.is_file():
						sub_path.unlink()
					elif sub_path.is_dir():
						sub_path.rmdir()
				self.old_checkpoint.rmdir()
			
			self.old_checkpoint = path
		
	def do_validation(self):
		assert self.model is not None and self.validation_dataloader is not None and self.compiled_model is not None

		log_rank_0(self.logger, logging.INFO, "Validating...")
		log_rank_0(self.logger, logging.INFO, f"Patch Dropout Rate: {self.model.patch_dropout}")

		self.compiled_model.eval()

		dataloader_iter = iter(self.validation_dataloader)
		total_loss = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)
		total_mae_loss = torch.tensor(0.0, device=self.device, requires_grad=False, dtype=torch.float32)
		keys = ('tags', 'reconstructed_tags')
		gpu_data_processing = GPUDataProcessing(mixup_alpha=0.0)

		results = {k: MetricCounters(num_classes=(self.num_tags if k == 'tags' or k == 'reconstructed_tags' else 1), device=self.device) for k in keys }

		# Set seed for reproducibility
		with torch.no_grad(), temprngstate(42):
			for batch in tqdm(dataloader_iter, disable=self.rank != 0, desc="Validation", dynamic_ncols=True):
				# Move to device
				batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

				# GPU side data augmentation
				batch = gpu_data_processing(batch)

				# Forward
				preds = self.compiled_model(batch, return_loss=True)
				loss = preds['loss']
				
				total_loss.add_(loss.detach())
				if 'mae_loss' in preds:
					total_mae_loss.add_(preds['mae_loss'].detach())
				
				for k in keys:
					if k not in preds:
						continue

					if k == 'tags' or k == 'reconstructed_tags':
						thresholded = torch.sigmoid(preds[k]).detach() > 0.5
						reference = batch['tags'] > 0.5
						results[k].true_positives.add_((thresholded & reference).sum(dim=0))
						results[k].false_positives.add_((thresholded & ~reference).sum(dim=0))
						results[k].true_negatives.add_((~thresholded & ~reference).sum(dim=0))
						results[k].false_negatives.add_((~thresholded & reference).sum(dim=0))
					else:
						thresholded = torch.argmax(preds[k].detach(), dim=1)
						results[k].true_positives.add_((thresholded == batch[k]).sum())
						results[k].false_positives.add_((thresholded != batch[k]).sum())
						results[k].true_negatives.add_((thresholded != batch[k]).sum())
						results[k].false_negatives.add_((thresholded == batch[k]).sum())
		
		# Gather across devices onto rank 0
		for k, v in results.items():
			torch.distributed.reduce(v.true_positives, dst=0, op=torch.distributed.ReduceOp.SUM)
			torch.distributed.reduce(v.false_positives, dst=0, op=torch.distributed.ReduceOp.SUM)
			torch.distributed.reduce(v.true_negatives, dst=0, op=torch.distributed.ReduceOp.SUM)
			torch.distributed.reduce(v.false_negatives, dst=0, op=torch.distributed.ReduceOp.SUM)
		
		total_loss = total_loss / len(self.validation_dataloader)
		torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.SUM)
		total_loss = total_loss / self.world_size

		total_mae_loss = total_mae_loss / len(self.validation_dataloader)
		torch.distributed.all_reduce(total_mae_loss, op=torch.distributed.ReduceOp.SUM)
		total_mae_loss = total_mae_loss / self.world_size
		
		# All other ranks are done
		if self.rank != 0:
			return
		
		metrics: dict[str, Union[int, float, torch.Tensor]] = {
			'validation/samples': self.global_samples_seen,
		}

		for k in ('tags', 'reconstructed_tags'):
			if not torch.any((results[k].true_positives + results[k].false_positives + results[k].true_negatives + results[k].false_negatives) > 0):
				continue

			accuracy = results[k].accuracy()
			precision = results[k].precision()
			recall = results[k].recall()
			f1 = results[k].f1()

			metrics[f"validation/{k}_accuracy"] = accuracy.item() if accuracy.numel() == 1 else accuracy
			metrics[f"validation/{k}_precision"] = precision.item() if precision.numel() == 1 else precision
			metrics[f"validation/{k}_recall"] = recall.item() if recall.numel() == 1 else recall
			metrics[f"validation/{k}_f1"] = f1.item() if f1.numel() == 1 else f1

			if accuracy.numel() > 1:
				metrics[f"validation/{k}_accuracy_mean"] = accuracy.mean()
				metrics[f"validation/{k}_precision_mean"] = precision.mean()
				metrics[f"validation/{k}_recall_mean"] = recall.mean()
				metrics[f"validation/{k}_f1_mean"] = f1.mean()

		metrics['validation/loss'] = total_loss.item()

		wandb.log(metrics, step=self.global_step)


@record
def main():
	logger = logging.getLogger(f'Process-{distributed_rank()}')
	logging.basicConfig(format='%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] - %(message)s')
	logger.setLevel(logging.INFO)

	if distributed_rank() == 0:
		# Parse configuration
		# Parse args
		config = parse_args_into_config(Config, logger)
		if config is None:
			torch.distributed.broadcast_object_list([None, None])
			return
		
		# Start
		wc = omegaconf.OmegaConf.to_container(config, resolve=True)
		assert isinstance(wc, dict)
		w = wandb.init(config=wc, project=config.wandb_project)
		assert w is not None
		with w:
			assert wandb.run is not None

			if wandb.run.resumed and config.resume is None:
				# Search for the folder with the highest number
				checkpoints = list(Path(config.output_dir / wandb.run.id).glob("samples_*"))
				checkpoints = [c.name for c in checkpoints if c.is_dir()]
				checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1]), reverse=True)
				if len(checkpoints) > 0:
					config.resume = Path(config.output_dir) / wandb.run.id / checkpoints[0]
					logger.info(f"WandB run resumed, loading latest checkpoint {config.resume}")
				else:
					logger.warning("WandB run resumed, but no checkpoints found")
			
			# Broadcast the config and run_id to all other processes
			torch.distributed.broadcast_object_list([config, wandb.run.id])

			logger.info("Rank 0 starting training...")
			trainer = MainTrainer(config=config, run_id=wandb.run.id, logger=logger)
			trainer.train()
	else:
		objects = [None, None]
		logger.info(f"Rank {distributed_rank()} waiting for config...")
		torch.distributed.broadcast_object_list(objects)
		config, run_id = objects

		if config is None or run_id is None:
			logger.info(f"Rank {distributed_rank()} exiting...")
			return
		
		logger.info(f"Rank {distributed_rank()} starting training...")
		trainer = MainTrainer(config=config, run_id=run_id, logger=logger)
		trainer.train()


if __name__ == "__main__":
	distributed_setup()
	torch.cuda.set_device(distributed_rank())
	main()
	distributed_cleanup()