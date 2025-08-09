import os
import torch
import pandas as pd
import glob
import numpy as np
from nflows import flows, transforms, distributions, utils
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import sys
sys.path.append("/home/koujing/ICCAD_2025/")
from Code.model.physical_layer.gaussian_layer import Gaussian_Layer
import random
import torch.nn as nn
from Code.Open_Source.Src.model import build_CNF, save_model


def load_dataset(x_tensor, w_tensor, error_tensor, die_tensor, train_ratio=0.8):
    """
    Wrap raw tensors into a TensorDataset and split into train/validation subsets.

    Args:
        x_tensor (Tensor): input tensor X.
        w_tensor (Tensor): weight tensor W.
        error_tensor (Tensor): Target error tensor to model with the flow.
        die_tensor (Tensor): Die index tensor.
        train_ratio (float): Proportion of samples used for training.

    Returns:
        (Subset, Subset): Train subset and validation subset.
    """
    # Order in the dataset: (x, w, die, error)
    dataset = TensorDataset(x_tensor, w_tensor, die_tensor, error_tensor)

    # Compute split sizes
    num_total = len(dataset)
    num_train = int(train_ratio * num_total)
    num_val = num_total - num_train

    # Random split keeps indices for each subset
    train_set, val_set = torch.utils.data.random_split(dataset, [num_train, num_val])

    return train_set, val_set


def train_model(
    model,
    x_tensor,
    w_tensor,
    error_tensor,
    die_tensor,
    model_save_dir,
    plot_save_dir,
    block_size,
    batch_size=1024,
    num_epochs=500,
    learning_rate=1e-4,
    save_interval=50,
    norm_params=None
):
    """
    Train the conditional normalizing flow model using negative log-likelihood loss.

    The context for the flow is built by concatenating [x, w, die].
    The target distribution is the error tensor.

    Args:
        model (nn.Module): Flow model with .log_prob(z, context=...).
        x_tensor, w_tensor, error_tensor, die_tensor (Tensor): Data tensors.
        model_save_dir (str): Directory to save checkpoints.
        plot_save_dir (str): Directory to save loss curves.
        block_size (tuple): Not used internally here; kept for compatibility/logging.
        batch_size (int): Training batch size.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Initial learning rate for Adam.
        save_interval (int): Save checkpoint every N epochs.
        norm_params (dict): Normalization parameters to store alongside checkpoints.

    Returns:
        model (nn.Module): Trained model (in-place updated).
        train_losses (list[float]): Per-epoch training NLL.
        val_losses (list[float]): Per-epoch validation NLL.
    """
    # Split full dataset into train/validation sets
    train_set, val_set = load_dataset(x_tensor, w_tensor, error_tensor, die_tensor)

    # DataLoaders: shuffle train, keep val deterministic
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=8
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16
    )

    # (Optional convenience tensors for direct validation access; not used further)
    val_x_tensor = val_set.dataset.tensors[0][val_set.indices]
    val_w_tensor = val_set.dataset.tensors[1][val_set.indices]
    val_die_tensor = val_set.dataset.tensors[2][val_set.indices]
    val_error_tensor = val_set.dataset.tensors[3][val_set.indices]

    # Choose device based on the model's parameters (assumes model.to(device) was called)
    device = next(model.parameters()).device

    # Optimizer and LR scheduler: ReduceLROnPlateau halves LR when val loss plateaus
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Save an initial checkpoint for reproducibility (epoch 0)
    save_model(
        model, optimizer, epoch=0, loss=0, norm_params=norm_params,
        model_save_dir=model_save_dir, filename="enhanced_nsf_model_epoch_0.pt"
    )

    train_losses, val_losses = [], []

    for epoch in range(1, num_epochs + 1):
        # --------------------
        # Training phase
        # --------------------
        model.train()
        train_epoch_losses = []

        for x_batch, w_batch, die_batch, error_batch in tqdm(
            train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"
        ):
            # Move mini-batch to the same device as the model
            x_batch = x_batch.to(device)
            w_batch = w_batch.to(device)
            die_batch = die_batch.to(device)
            error_batch = error_batch.to(device)

            optimizer.zero_grad()

            # Build conditional context: [x, w, die]
            condition = torch.cat([x_batch, w_batch, die_batch], dim=1)

            # Negative log-likelihood loss: minimize -log p(error | context)
            loss = -model.log_prob(error_batch, context=condition).mean()

            # Backpropagation
            loss.backward()

            # Gradient clipping to stabilize training for deep flows
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            train_epoch_losses.append(loss.item())

        # --------------------
        # Validation phase (no grad)
        # --------------------
        model.eval()
        val_epoch_losses = []

        with torch.no_grad():
            for x_batch, w_batch, die_batch, error_batch in tqdm(
                val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"
            ):
                x_batch = x_batch.to(device)
                w_batch = w_batch.to(device)
                die_batch = die_batch.to(device)
                error_batch = error_batch.to(device)

                condition = torch.cat([x_batch, w_batch, die_batch], dim=1)
                loss = -model.log_prob(error_batch, context=condition).mean()
                val_epoch_losses.append(loss.item())

        # Aggregate average losses for this epoch
        avg_train_loss = np.mean(train_epoch_losses)
        avg_val_loss = np.mean(val_epoch_losses)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        # Step LR scheduler on validation loss
        scheduler.step(avg_val_loss)

        # Periodically save checkpoints (contains model/optimizer/norm params)
        if epoch % save_interval == 0 or epoch == num_epochs:
            save_model(
                model, optimizer, epoch, avg_val_loss, norm_params,
                model_save_dir, f"enhanced_nsf_model_epoch_{epoch}.pt"
            )

    # --------------------
    # Plot & save loss curves
    # --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Negative Log-Likelihood Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # NOTE: save to plot_save_dir (was model_save_dir before)
    os.makedirs(plot_save_dir, exist_ok=True)
    loss_plot_path = os.path.join(plot_save_dir, "enhanced_training_loss.png")
    plt.savefig(loss_plot_path)
    plt.close()
    print(f"Training/validation loss plot saved to {loss_plot_path}")

    return model, train_losses, val_losses


def main():
    """
    End-to-end training entry:
      1) Load data from disk
      2) Normalize inputs (x, w)
      3) Build CNF model and move to device
      4) Train model and save artifacts
    """
    from Code.Open_Source.Src.data import load_readback_data_multidie
    import torch.multiprocessing as mp
    from Code.Open_Source.Src.utils import set_seed, normalize
    import warnings

    mp.set_start_method('spawn', force=True)
    set_seed(42)
    warnings.filterwarnings("ignore")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Directory where the block data are stored
    data_dir = "/home/koujing/ICCAD_2025/Code/data/readback_new/block_4x32"

    # ------------------------------------------------------------
    # 1) Load raw tensors
    # ------------------------------------------------------------
    x_tensor, w_tensor, error_tensor, die_tensor = load_readback_data_multidie(
        base_dir=data_dir,
        block_size=(4, 32),
        max_rows_per_file=1000,
        num_workers=16
    )

    # ------------------------------------------------------------
    # 2) Normalize inputs (keep stats for inference)
    # ------------------------------------------------------------
    x_norm, w_norm, norm_params = normalize(x_tensor, w_tensor)

    # ------------------------------------------------------------
    # 3) Build CNF model with physical priors and send to device
    # ------------------------------------------------------------
    model = build_CNF(
        norm_params,
        condition_dim=135,
        target_dim=128,
        n_die=3,
        num_transforms=8,
        hidden_features=512,
        max_rescale=2.0,
        # Physical prior coefficients (trainable=True allows fine-tuning)
        a=0.107830, b=0.287604, c=0.175547, d=0.039209,
        alpha_1=0.000010, alpha_2=0.028675,
        beta_1=0.000371, beta_2=0.013130,
        trainable=True,
    ).to(device)

    # Print model size as a quick sanity check
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # ------------------------------------------------------------
    # 4) Train
    # ------------------------------------------------------------
    model, train_losses, val_losses = train_model(
        model,
        x_norm,
        w_norm,
        error_tensor,
        die_tensor,
        batch_size=120000,
        num_epochs=200,
        learning_rate=1e-4,
        block_size=(4, 32),
        model_save_dir='/home/koujing/ICCAD_2025/open_source_model_test',
        plot_save_dir='/home/koujing/ICCAD_2025/open_source_plots_test',
        save_interval=10,
        norm_params=norm_params
    )

    print("Enhanced training and visualization complete.")


if __name__ == "__main__":
    main()
