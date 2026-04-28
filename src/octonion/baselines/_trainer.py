"""Training utility with full observability, checkpointing, and Optuna HP search.

Provides:
- seed_everything: Deterministic seeding for all RNGs
- train_model: Complete training loop with TensorBoard logging, gradient stats,
  VRAM monitoring, early stopping, LR warmup, AMP, and graceful shutdown
- evaluate: Model evaluation for classification tasks
- save_checkpoint / load_checkpoint: Full training state persistence
- run_optuna_study: Bayesian hyperparameter search via Optuna
"""

from __future__ import annotations

import json
import logging
import os
import random
import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from octonion.baselines._config import AlgebraType, TrainConfig

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Seeds Python, NumPy, and PyTorch RNGs. Does NOT set
    torch.backends.cudnn.deterministic = True (performance over
    bit-exactness per CONTEXT.md).

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_model_module(model: nn.Module) -> nn.Module:
    """Unwrap DDP wrapper if present."""
    if hasattr(model, "module"):
        inner = model.module
        assert isinstance(inner, nn.Module)
        return inner
    return model


def _wrap_manifold_params(
    model: nn.Module,
    algebra_type: AlgebraType,
    manifold_type: str = "sphere",
) -> nn.Module:
    """Replace algebra weight parameters with geoopt ManifoldParameter for Riemannian optimization.

    Only wraps weights in algebra-specific layers (not bias, BN, projection layers).

    Args:
        model: The network to wrap.
        algebra_type: Determines which manifold dimensions to use.
        manifold_type: "sphere" for S^{dim-1} (unit-norm per algebra element),
                       or "stiefel" for Stiefel manifold (orthogonality constraint
                       on weight matrices). Per user decision: try BOTH.

    Manifold assignments by algebra and manifold_type:
      - REAL: Euclidean (no constraint, regardless of manifold_type)
      - COMPLEX + sphere: geoopt.Sphere() on pairs (S^1)
      - QUATERNION + sphere: geoopt.Sphere() on quads (S^3)
      - OCTONION + sphere: geoopt.Sphere() on octets (S^7)
      - COMPLEX/QUATERNION/OCTONION + stiefel: geoopt.Stiefel() on weight matrices

    Returns:
        The model with manifold-wrapped parameters (modified in-place).
    """
    import geoopt

    if algebra_type in (AlgebraType.REAL,):
        return model  # Real: Euclidean, no manifold constraint

    if manifold_type == "sphere":
        manifold = geoopt.Sphere()
    elif manifold_type == "stiefel":
        manifold = geoopt.Stiefel()
    else:
        raise ValueError(f"Unknown manifold_type: {manifold_type}")

    # Walk the model and wrap algebra-specific weight parameters.
    # Strategy: wrap parameters in modules that are algebra layers
    # (OctonionDenseLinear, QuaternionLinear, ComplexLinear, PHM8Linear,
    # DenseMixingLinear) but NOT input_proj, output_proj, or BN modules.
    _skip_module_patterns = ("bn", "norm", "output_proj", "input_proj")
    _skip_param_patterns = ("bias",)

    # Import algebra layer types for identification
    from octonion.baselines._algebra_linear import (
        ComplexLinear,
        OctonionDenseLinear,
        QuaternionLinear,
    )

    _algebra_layer_types: tuple[type[nn.Module], ...] = (
        ComplexLinear, QuaternionLinear, OctonionDenseLinear,
    )
    try:
        from octonion.baselines._phm_linear import PHM8Linear
        _algebra_layer_types = (*_algebra_layer_types, PHM8Linear)
    except ImportError:
        pass
    try:
        from octonion.baselines._dense_mixing import DenseMixingLinear
        _algebra_layer_types = (*_algebra_layer_types, DenseMixingLinear)
    except ImportError:
        pass

    for mod_name, module in model.named_modules():
        # Skip BN and projection modules by name
        if any(skip in mod_name.lower() for skip in _skip_module_patterns):
            continue

        # For algebra layers, wrap their weight parameters
        if isinstance(module, _algebra_layer_types):
            # For OctonionDenseLinear, weights are in a ParameterList
            if hasattr(module, "weights") and isinstance(module.weights, nn.ParameterList):
                for idx in range(len(module.weights)):
                    param = module.weights[idx]
                    if manifold_type == "stiefel" and (param.dim() < 2 or param.shape[-2] < param.shape[-1]):
                        continue
                    try:
                        mp = geoopt.ManifoldParameter(param.data.clone(), manifold=manifold)
                        module.weights[idx] = mp
                    except Exception:
                        pass
            else:
                # For other algebra layers (ComplexLinear, QuaternionLinear, etc.)
                for pname, param in list(module.named_parameters(recurse=False)):
                    if any(skip in pname.lower() for skip in _skip_param_patterns):
                        continue
                    if manifold_type == "stiefel" and (param.dim() < 2 or param.shape[-2] < param.shape[-1]):
                        continue
                    try:
                        mp = geoopt.ManifoldParameter(param.data.clone(), manifold=manifold)
                        setattr(module, pname, mp)
                    except Exception:
                        pass

    return model


def _build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Build optimizer from config.

    Args:
        model: Model whose parameters to optimize.
        config: Training config with optimizer name, lr, and weight_decay.

    Returns:
        Configured optimizer instance.

    Raises:
        ValueError: If optimizer name is not recognized.
    """
    params = model.parameters()
    name = config.optimizer.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=config.lr, weight_decay=config.weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=config.lr, weight_decay=config.weight_decay,
            momentum=0.9, nesterov=getattr(config, "nesterov", False),
        )
    elif name == "lbfgs":
        return torch.optim.LBFGS(
            params, lr=config.lr, max_iter=20, line_search_fn="strong_wolfe",
        )
    elif name == "riemannian_adam":
        import geoopt
        return geoopt.optim.RiemannianAdam(  # type: ignore[no-any-return]
            params, lr=config.lr, weight_decay=config.weight_decay,
        )
    elif name == "shampoo":
        from pytorch_optimizer import Shampoo  # type: ignore[attr-defined]
        return Shampoo(params, lr=config.lr, weight_decay=config.weight_decay)
    else:
        raise ValueError(
            f"Unknown optimizer: {config.optimizer!r}. "
            f"Use 'adam', 'adamw', 'sgd', 'lbfgs', 'riemannian_adam', or 'shampoo'."
        )


def _build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainConfig
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build LR scheduler from config.

    Args:
        optimizer: Optimizer to schedule.
        config: Training config with scheduler name and epochs.

    Returns:
        Configured LR scheduler instance.

    Raises:
        ValueError: If scheduler name is not recognized.
    """
    name = config.scheduler.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, config.epochs // 3))
    elif name == "step_cifar":
        # Matches Gaudet & Maida 2018 / Trabelsi 2018 CIFAR schedule:
        # LR=lr for warmup (epochs 0-9), LR=lr*10 for epochs 10-119,
        # LR=lr at epoch 120, LR=lr/10 at epoch 150.
        #
        # Implemented as a MultiStepLR starting from the peak LR (lr*10),
        # with milestones at epochs 120 and 150 dividing by 10 each time.
        # The warmup from lr to lr*10 is handled separately in the training
        # loop (see warmup logic in train_model).
        #
        # IMPORTANT: The scheduler is only stepped after warmup_epochs
        # (i.e., starting at real epoch 10). To align milestones with real
        # epoch numbers, we subtract warmup_epochs from the milestones.
        # E.g., warmup_epochs=10 -> milestones=[110, 140] so the scheduler
        # triggers at real epochs 120 and 150.
        warmup = getattr(config, "warmup_epochs", 10)
        peak_lr = config.lr * 10  # e.g., 0.01 * 10 = 0.1
        for pg in optimizer.param_groups:
            pg["lr"] = peak_lr
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[120 - warmup, 150 - warmup], gamma=0.1
        )
    elif name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.5
        )
    elif name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    else:
        raise ValueError(
            f"Unknown scheduler: {config.scheduler!r}. "
            f"Use 'cosine', 'step', 'step_cifar', 'plateau', or 'none'."
        )


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: str | torch.device,
    loss_fn: nn.Module | None = None,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    For classification (CrossEntropyLoss): returns (avg_loss, accuracy).
    For regression (MSELoss, etc.): returns (avg_loss, 0.0) since accuracy
    is not meaningful for regression tasks.

    Args:
        model: Model to evaluate.
        loader: DataLoader for evaluation data.
        device: Device to run on.
        loss_fn: Loss function (defaults to CrossEntropyLoss).

    Returns:
        Tuple of (average loss, accuracy).
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            if is_classification:
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
            total += inputs.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1) if is_classification else 0.0
    return avg_loss, accuracy


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_val_loss: float,
    metrics: dict[str, Any],
) -> None:
    """Save full training state to a checkpoint file.

    Handles both DDP-wrapped and unwrapped models.

    Args:
        path: File path to save checkpoint.
        model: Model (may be DDP-wrapped).
        optimizer: Optimizer with state.
        scheduler: LR scheduler with state.
        epoch: Current epoch number.
        best_val_loss: Best validation loss seen so far.
        metrics: Training metrics dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "model_state_dict": _get_model_module(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "metrics": metrics,
    }
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict[str, Any]:
    """Load checkpoint and restore training state.

    Handles both DDP-wrapped and unwrapped models.

    Args:
        path: File path to load checkpoint from.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into (optional).
        scheduler: LR scheduler to restore state into (optional).

    Returns:
        Metadata dict with epoch, best_val_loss, metrics.
    """
    checkpoint = torch.load(path, weights_only=False)

    _get_model_module(model).load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
        "metrics": checkpoint.get("metrics", {}),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    config: TrainConfig,
    output_dir: str,
    device: str | torch.device = "cuda",
    loss_fn: nn.Module | None = None,
) -> dict[str, Any]:
    """Complete training loop with full observability.

    Features:
    - LR warmup (linear from 0 to target_lr over warmup_epochs)
    - AMP (optional, via config.use_amp)
    - Gradient statistics logging (norm mean, max, variance)
    - VRAM peak monitoring (when CUDA available)
    - TensorBoard logging (loss, accuracy, LR, gradient stats, VRAM)
    - Checkpointing (every config.checkpoint_every epochs)
    - Early stopping (after config.early_stopping_patience epochs)
    - Graceful shutdown (SIGINT saves checkpoint)

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training hyperparameters.
        output_dir: Directory for checkpoints, TensorBoard logs.
        device: Device to train on ('cpu', 'cuda', etc.).
        loss_fn: Loss function (defaults to CrossEntropyLoss).

    Returns:
        Dict with train_losses, val_losses, val_accuracies, best_val_acc,
        best_val_loss, total_time_seconds, epochs_trained, early_stopped,
        lr_history.
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Enable cuDNN autotuner for fixed-size inputs (CIFAR batch_size=128, drop_last=True).
    # Safe because input sizes are constant throughout training.
    # Per project policy: "seed-controlled but not CUDA deterministic".
    if str(device).startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Optional torch.compile for potential kernel fusion speedup.
    # Gated by config flag; only applied on CUDA devices (ROCm support is
    # experimental and may fail for certain model patterns).
    if getattr(config, "use_compile", False) and str(device).startswith("cuda"):
        try:
            model = torch.compile(model, backend="inductor", mode="default")  # type: ignore[assignment]
            logger.info("torch.compile enabled (inductor backend)")
        except Exception as e:
            logger.warning("torch.compile failed, falling back to eager mode: %s", e)

    # Build optimizer and scheduler
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)

    # AMP setup
    use_amp = config.use_amp and device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]

    # TensorBoard — typed as Any because SummaryWriter methods (add_scalar, close)
    # are untyped in PyTorch stubs.
    writer: Any = SummaryWriter(log_dir=output_dir)  # type: ignore[no-untyped-call]

    # LR warmup configuration
    warmup_epochs = config.warmup_epochs
    is_step_cifar = config.scheduler.lower() == "step_cifar"

    if is_step_cifar:
        # step_cifar: warmup holds LR at config.lr (e.g., 0.01) for warmup_epochs,
        # then jumps to peak LR (config.lr * 10 = 0.1) when warmup ends.
        # The MultiStepLR scheduler already has peak LR set; during warmup we
        # override to config.lr. After warmup, we let the scheduler control LR.
        warmup_lr = config.lr  # Hold at base LR during warmup
        target_lr = config.lr * 10  # Peak LR after warmup
        if warmup_epochs > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr
    else:
        # Default: linear warmup from 0 to config.lr
        target_lr = config.lr
        if warmup_epochs > 0:
            for pg in optimizer.param_groups:
                pg["lr"] = 0.0

    # Tracking
    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []
    lr_history: list[float] = []
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    early_stopped = False
    interrupted = False

    # Graceful shutdown handler
    original_handler = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        ckpt_path = os.path.join(output_dir, "checkpoint_interrupted.pt")
        save_checkpoint(
            ckpt_path, model, optimizer, scheduler, epoch, best_val_loss,
            {"train_losses": train_losses, "val_losses": val_losses},
        )
        logger.info(f"Interrupted: checkpoint saved to {ckpt_path}")
        signal.signal(signal.SIGINT, original_handler)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    start_time = time.time()

    try:
        for epoch in range(config.epochs):
            # Compute current LR based on warmup
            if warmup_epochs > 0 and epoch < warmup_epochs:
                if is_step_cifar:
                    # step_cifar: hold at warmup_lr (config.lr) during warmup
                    for pg in optimizer.param_groups:
                        pg["lr"] = warmup_lr
                else:
                    # Default: linear warmup from 0 to target_lr
                    warmup_factor = (epoch + 1) / max(1, warmup_epochs)
                    current_lr = target_lr * warmup_factor
                    for pg in optimizer.param_groups:
                        pg["lr"] = current_lr
            elif warmup_epochs > 0 and epoch == warmup_epochs:
                # Restore target LR at end of warmup
                for pg in optimizer.param_groups:
                    pg["lr"] = target_lr

            # Record current LR
            lr_history.append(optimizer.param_groups[0]["lr"])

            # ── Training ──
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
            amp_device_type = "cuda" if str(device).startswith("cuda") else "cpu"

            for batch in train_loader:
                inputs = batch[0].to(device, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)

                if is_lbfgs:
                    # LBFGS requires a closure that re-evaluates the model.
                    # Bind inputs/targets as default args so the closure captures
                    # this iteration's values defensively (LBFGS only invokes
                    # synchronously today, but binding is cheap insurance).
                    batch_loss_value = [0.0]

                    def closure(
                        inputs: torch.Tensor = inputs,
                        targets: torch.Tensor = targets,
                    ) -> torch.Tensor:
                        optimizer.zero_grad(set_to_none=True)
                        with torch.amp.autocast(amp_device_type, enabled=use_amp):  # type: ignore[attr-defined] # noqa: B023 — loop-invariant
                            outputs = model(inputs)
                            loss = loss_fn(outputs, targets)
                        loss.backward()
                        batch_loss_value[0] = loss.item()  # noqa: B023 — mutation via list index, not rebind
                        return loss  # type: ignore[no-any-return]

                    optimizer.step(closure)  # type: ignore[arg-type]
                    epoch_loss += batch_loss_value[0]
                else:
                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(amp_device_type, enabled=use_amp):  # type: ignore[attr-defined]
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets)

                    scaler.scale(loss).backward()

                    # Gradient clipping (if configured)
                    grad_clip = getattr(config, "gradient_clip_norm", 0.0)
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    scaler.step(optimizer)
                    scaler.update()

                    epoch_loss += loss.item()

                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            train_losses.append(avg_train_loss)

            # ── Gradient statistics ──
            grad_norms: list[float] = []
            for _name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norms.append(param.grad.norm().item())

            if grad_norms:
                grad_mean = sum(grad_norms) / len(grad_norms)
                grad_max = max(grad_norms)
                grad_var = (
                    sum((g - grad_mean) ** 2 for g in grad_norms) / len(grad_norms)
                )
                writer.add_scalar("Grad/norm_mean", grad_mean, epoch)
                writer.add_scalar("Grad/norm_max", grad_max, epoch)
                writer.add_scalar("Grad/norm_var", grad_var, epoch)

            # ── BN condition number monitoring ──
            max_cond = 0.0
            for _name, mod in model.named_modules():
                if hasattr(mod, "last_cond"):
                    c = mod.last_cond.item()
                    if c > max_cond:
                        max_cond = c
            if max_cond > 0:
                writer.add_scalar("BN/max_cond_number", max_cond, epoch)
                if max_cond > 1e4:
                    logger.warning(
                        f"Epoch {epoch}: BN condition number {max_cond:.1f} "
                        "— whitening may lose precision"
                    )

            # ── VRAM monitoring ──
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                vram_peak = torch.cuda.max_memory_allocated() / 1e6
                writer.add_scalar("VRAM/peak_MB", vram_peak, epoch)

            # ── Validation ──
            # Use same autocast context as training to avoid torch._dynamo
            # recompile churn (GLOBAL_STATE changed: grad_mode autocast).
            with torch.amp.autocast(amp_device_type, enabled=use_amp):  # type: ignore[attr-defined]
                val_loss, val_acc = evaluate(model, val_loader, device, loss_fn)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # ── Progress logging (stdout) ──
            elapsed = time.time() - start_time
            epoch_time = elapsed / (epoch + 1)
            remaining = epoch_time * (config.epochs - epoch - 1)
            remaining_min = remaining / 60
            is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)

            def _fmt_loss(v: float) -> str:
                return f"{v:.4f}" if v >= 1e-4 else f"{v:.4e}"

            if is_classification:
                err_pct = (1 - val_acc) * 100
                val_metric = f"val_acc={val_acc:.4f} ({err_pct:.2f}% err)"
            else:
                val_metric = f"val_loss={_fmt_loss(val_loss)}"

            print(
                f"[{epoch + 1:3d}/{config.epochs}] "
                f"train_loss={_fmt_loss(avg_train_loss)}  "
                f"{val_metric}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                f"~{remaining_min:.0f}min left",
                flush=True,
            )

            # ── TensorBoard logging ──
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # ── Scheduler step ──
            if epoch >= warmup_epochs:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # ── Early stopping ──
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.early_stopping_patience:
                    early_stopped = True
                    break

            # ── Checkpointing ──
            if (epoch + 1) % config.checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_epoch{epoch + 1}.pt")
                save_checkpoint(
                    ckpt_path, model, optimizer, scheduler, epoch, best_val_loss,
                    {"train_losses": train_losses, "val_losses": val_losses},
                )

    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)
        writer.close()

    total_time = time.time() - start_time

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time_seconds": total_time,
        "epochs_trained": len(train_losses),
        "early_stopped": early_stopped,
        "lr_history": lr_history,
    }


def run_optuna_study(
    model_builder_fn: Callable[[AlgebraType], nn.Module],
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    algebra: AlgebraType,
    n_trials: int = 50,
    study_name: str = "hp_search",
    output_dir: str = "experiments/optuna",
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    """Run Bayesian hyperparameter search via Optuna.

    Creates an Optuna study that searches over lr, weight_decay, optimizer,
    scheduler, batch_size, and gradient_clip. Uses MedianPruner for early
    stopping of bad trials.

    Args:
        model_builder_fn: Callable that takes AlgebraType and returns nn.Module.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        algebra: Algebra type for the model.
        n_trials: Number of Optuna trials to run.
        study_name: Name for the Optuna study.
        output_dir: Directory for saving results.
        device: Device to train on.

    Returns:
        Dict with best_params, best_value, n_trials, study_name.
    """
    import optuna

    # Suppress Optuna logs during search
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=2),
    )

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
        scheduler_name = trial.suggest_categorical("scheduler", ["cosine", "step", "plateau"])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        trial.suggest_float("gradient_clip", 0.0, 5.0)

        # Build config with reduced epochs for search
        config = TrainConfig(
            epochs=20,
            lr=lr,
            optimizer=optimizer_name,
            scheduler=scheduler_name,
            weight_decay=weight_decay,
            early_stopping_patience=5,
            warmup_epochs=2,
            use_amp=False,
            checkpoint_every=100,  # Don't checkpoint during search
            seed=42,
            batch_size=batch_size,
        )

        # Build model
        model = model_builder_fn(algebra)

        # Train
        trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
        try:
            result = train_model(
                model, train_loader, val_loader, config, trial_dir, device=device
            )
        except Exception as e:
            logger.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")

        # Report intermediate values for pruning
        for i, val_loss in enumerate(result["val_losses"]):
            trial.report(val_loss, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(result["best_val_loss"])

    study.optimize(objective, n_trials=n_trials)

    # Extract results
    best_trial = study.best_trial
    result = {
        "best_params": best_trial.params,
        "best_value": best_trial.value,
        "n_trials": len(study.trials),
        "study_name": study_name,
    }

    # Save results to JSON
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{study_name}_results.json")
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
