"""
Training pipeline for Gearhead model.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_device_info():
    """
    Detect available compute devices (CUDA, ROCm, MPS, or CPU).

    Returns:
        tuple: (device_type, device_name, is_available)
    """
    # Check for ROCm (AMD GPUs)
    if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        device_name = torch.cuda.get_device_name(0)
        return 'rocm', device_name, True
    # Check for CUDA (NVIDIA GPUs)
    elif torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return 'cuda', device_name, True
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # Detect chip type
        import platform
        chip = platform.processor() or platform.machine()
        return 'mps', f'Apple Silicon ({chip})', True
    # Fallback to CPU
    else:
        return 'cpu', 'CPU', True


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Data
    train_data_path: str
    val_data_path: Optional[str] = None
    max_seq_length: int = 2048

    # Training hyperparameters
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # Optimization
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8

    # Checkpointing
    output_dir: str = "./outputs"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3

    # Logging
    use_wandb: bool = False
    wandb_project: str = "gearhead"
    wandb_run_name: Optional[str] = None

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    fp16: bool = False

    # ROCm-specific optimizations
    use_rocm: bool = False  # Auto-detected if not explicitly set
    rocm_optimize: bool = True  # Enable ROCm-specific optimizations


class GearheadTrainer:
    """
    Trainer for Gearhead model.

    Handles training loop, evaluation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset,
        tokenizer,
        config: TrainingConfig,
        val_dataset=None,
        data_collator=None,
    ):
        """
        Initialize trainer.

        Args:
            model: GearheadModel instance
            train_dataset: Training dataset
            tokenizer: Tokenizer instance
            config: Training configuration
            val_dataset: Optional validation dataset
            data_collator: Optional data collator
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.data_collator = data_collator

        # Detect device type
        device_type, device_name, _ = get_device_info()
        self.device_type = device_type
        self.device_name = device_name

        # Auto-detect ROCm if not explicitly set
        if device_type == 'rocm':
            self.config.use_rocm = True

        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # Ensure model parameters are contiguous for ROCm compatibility
        if self.device_type == 'rocm':
            self._make_model_contiguous()

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for memory efficiency")

        # Setup optimization
        self.optimizer = self._create_optimizer()

        # Setup mixed precision scaler
        # MPS doesn't support GradScaler yet, use autocast only
        if config.fp16:
            if device_type in ['cuda', 'rocm']:
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.scaler = None  # CPU and MPS use autocast without scaler

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup logging
        if config.use_wandb and HAS_WANDB:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config),
            )

        # Training state
        self.global_step = 0
        self.epoch = 0

        # ROCm-specific optimizations
        if self.config.use_rocm and self.config.rocm_optimize:
            self._apply_rocm_optimizations()

    def _make_model_contiguous(self):
        """Ensure all model parameters are contiguous in memory for ROCm."""
        print("Making model parameters contiguous for ROCm compatibility...")
        for name, param in self.model.named_parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        print("Model parameters made contiguous")

    def _apply_rocm_optimizations(self):
        """Apply ROCm-specific optimizations for AMD GPUs."""
        print(f"Applying ROCm optimizations for {self.device_name}")

        # Enable TF32 tensor cores on compatible AMD GPUs (CDNA2+)
        # ROCm 5.0+ supports this for matrix operations
        if hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Set memory allocator settings for better memory efficiency
        if hasattr(torch.cuda, 'memory'):
            # Use native allocator for better ROCm performance
            os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')

        print("ROCm optimizations applied")

    def _create_optimizer(self):
        """Create AdamW optimizer with weight decay."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # No weight decay for biases and layer norms
            if "bias" in name or "ln" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
        )

        return optimizer

    def _get_lr_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        from torch.optim.lr_scheduler import LambdaLR

        def lr_lambda(current_step: int):
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) / float(
                    max(1, num_training_steps - self.config.warmup_steps)
                ),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self):
        """Run training loop."""
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            num_workers=0,
        )

        val_loader = None
        if self.val_dataset is not None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                num_workers=0,
            )

        # Calculate total steps
        num_update_steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        num_training_steps = num_update_steps_per_epoch * self.config.num_epochs

        # Create scheduler
        scheduler = self._get_lr_scheduler(num_training_steps)

        # Training loop
        self.model.train()
        total_loss = 0.0

        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total training steps: {num_training_steps}")
        print(f"Device: {self.device} ({self.device_type.upper()}: {self.device_name})")
        if self.config.use_rocm:
            print(f"ROCm optimizations: {'Enabled' if self.config.rocm_optimize else 'Disabled'}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for step, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass with appropriate autocast for device
                if self.config.fp16:
                    if self.device_type in ['cuda', 'rocm']:
                        with torch.cuda.amp.autocast():
                            logits, loss = self.model(**batch)
                    elif self.device_type == 'mps':
                        # MPS uses torch.amp.autocast
                        with torch.amp.autocast('cpu', dtype=torch.float16):
                            logits, loss = self.model(**batch)
                    else:
                        logits, loss = self.model(**batch)
                else:
                    logits, loss = self.model(**batch)

                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps

                # Backward pass
                if self.config.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total_loss += loss.item()
                epoch_loss += loss.item()

                # Update weights
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.fp16:
                        self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                        self.optimizer.step()

                    scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    # Update progress bar
                    pbar.set_postfix({"loss": loss.item() * self.config.gradient_accumulation_steps})

                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = total_loss / self.config.logging_steps
                        self._log_metrics({
                            "train/loss": avg_loss,
                            "train/learning_rate": scheduler.get_last_lr()[0],
                            "train/epoch": epoch,
                        })
                        total_loss = 0.0

                    # Evaluation
                    if val_loader and self.global_step % self.config.eval_steps == 0:
                        val_loss = self.evaluate(val_loader)
                        self._log_metrics({"eval/loss": val_loss})
                        self.model.train()

                    # Checkpointing
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint()

            # End of epoch
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint(final=True)
        print("Training completed!")

    def evaluate(self, val_loader):
        """Run evaluation on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.config.fp16:
                    with torch.cuda.amp.autocast():
                        logits, loss = self.model(**batch)
                else:
                    logits, loss = self.model(**batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Validation loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, final: bool = False):
        """Save model checkpoint."""
        checkpoint_name = "final_model" if final else f"checkpoint-{self.global_step}"
        checkpoint_path = Path(self.config.output_dir) / checkpoint_name

        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_path / "model.pt",
        )

        # Save config
        torch.save(
            self.model.config,
            checkpoint_path / "config.pt",
        )

        # Save training state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
                "global_step": self.global_step,
            },
            checkpoint_path / "trainer_state.pt",
        )

        print(f"Checkpoint saved to {checkpoint_path}")

    def _log_metrics(self, metrics: dict):
        """Log metrics to wandb if enabled."""
        if self.config.use_wandb and HAS_WANDB:
            wandb.log(metrics, step=self.global_step)
