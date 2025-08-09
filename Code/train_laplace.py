import torch
from tqdm import tqdm

def train_model(model, y_train, error_train, epochs=1000, lr=0.01):
    """
    Train the output layer using negative log-likelihood loss.

    Args:
        model (torch.nn.Module): The model to be trained (e.g., Output_Layer for Laplace).
        y_train (torch.Tensor): Input y values for training.
        error_train (torch.Tensor): Target error values corresponding to y_train.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.

    Returns:
        tuple:
            model (torch.nn.Module): The trained model.
            losses (list): List of per-epoch loss values.

    Notes:
        - Uses Adam optimizer and ReduceLROnPlateau scheduler to adapt learning rate.
        - Loss is computed as the mean NLL over the batch.
        - Prints final parameters after training.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200, verbose=True
    )
    losses = []

    for epoch in tqdm(range(epochs), desc=f"Training {model.distribution}"):
        optimizer.zero_grad()
        mean, sigma = model(y_train)
        loss = model.negative_log_likelihood(mean, sigma, error_train).mean()

        if torch.isnan(loss):
            print(f"[{model.distribution}] NaN loss detected, stopping.")
            break

        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        losses.append(loss.item())

        if epoch % 100 == 0:
            tqdm.write(f"[{model.distribution}] Epoch {epoch}: Loss = {loss.item():.6f}")

    # Output final parameters
    coeffs_val = model.coeffs.detach().cpu().numpy().tolist()
    k_val = float(torch.exp(model.log_k).item())
    b_val = float(torch.exp(model.log_b).item())
    print(f"[{model.distribution}] Final parameters:")
    print(f"  Coeffs = {[c for c in coeffs_val]}")
    print(f"  k = {k_val}")
    print(f"  b = {b_val}")

    print(f"[{model.distribution}] Training completed")
    return model, losses


def main():
    """
    Main training script to load real data, filter it, and train the Laplace Output Layer.
    """
    from Code.Open_Source.Src.data import load_infer_tensors_from_directory
    from Code.Open_Source.Src.layers import Output_Layer

    # ===== Configuration =====
    config = {
        "data_dir": "/home/koujing/ICCAD_2025/Code/data/infer_opensource",
        "y_range": (-100, 100),      
        "epochs": 5000,             
        "lr": 5e-3,                  
        "num_workers": 16,         
        "init_mean_coeffs": (0.1, 0.1, 0.1),
        "init_std_scale": 1.0,
        "init_std_bias": 1.0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Load real dataset =====
    y_data, error_data = load_infer_tensors_from_directory(
        directory=config["data_dir"],
        flatten=True,
        device=device,
        num_workers=config["num_workers"]
    )

    # Filter dataset to only include y in the specified range
    y_min, y_max = config["y_range"]
    train_mask = (y_data >= y_min) & (y_data <= y_max)
    y_train = y_data[train_mask]
    error_train = error_data[train_mask]
    print(f"Train samples: {y_train.numel()} within y ∈ [{y_min}, {y_max}]")

    # ===== Initialize Laplace Output Layer =====
    laplace_model = Output_Layer(
        device=device,
        trainable=True,
        init_mean_coeffs=config["init_mean_coeffs"],
        init_std_scale=config["init_std_scale"],
        init_std_bias=config["init_std_bias"]
    ).to(device)

    # ===== Train model =====
    train_model(
        model=laplace_model,
        y_train=y_train,
        error_train=error_train,
        epochs=config["epochs"],
        lr=config["lr"]
    )

if __name__ == "__main__":
    main()
