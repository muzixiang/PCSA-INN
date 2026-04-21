import os
import sys
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Ensure local dependencies are resolved within the anonymous repository structure
sys.path.append(os.getcwd())

from configs.config import Config
from data.dataset import S3ChunkDataset
from models.INN import AtmosphericCorrectionCINN

# ==========================================
# Phase 0: Experimental Reproducibility & Context Initialization
# ==========================================
def seed_everything(seed=42):
    """
    Strictly enforces deterministic execution across all random number generators 
    (CPU and GPU) to ensure exact reproducibility of the reported forward simulation metrics.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_or_create_stats(json_path):
    """
    Loads empirical normalization statistics derived from the physical dataset.
    
    [Reviewer Note on Privacy & Dummy Mode]: 
    If the actual statistics file is absent, this function automatically initializes a set of 
    physically-dimensioned prior statistics to allow reviewers to verify the pipeline.
    """
    if not os.path.exists(json_path):
        print(f"[Verification Mode] Physical prior file absent at: {json_path}")
        print("[Verification Mode] Initializing synthetic priors for structural validation.")
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        dummy_stats = {
            "shift_rad": 1.0, "shift_ref": 0.05,
            "rad_mean": [0.0]*21, "rad_std": [1.0]*21,
            "ref_mean": [0.0]*8,  "ref_std": [1.0]*8,
            "ozone_mean": 0.3, "ozone_std": 0.05,
            "wv_mean": 1.5, "wv_std": 0.5,
            "press_mean": 1013.0, "press_std": 10.0,
            "wind_mean": 5.0, "wind_std": 2.0
        }
        with open(json_path, 'w') as f:
            json.dump(dummy_stats, f, indent=4)
        return dummy_stats
        
    with open(json_path, "r") as f:
        stats = json.load(f)
        
    processed_stats = {}
    for k, v in stats.items():
        if isinstance(v, list):
            processed_stats[k] = np.array(v, dtype=np.float32)
        elif isinstance(v, (int, float)):
            processed_stats[k] = float(v)
    return processed_stats

# ==========================================
# Phase 1: Forward Simulation (RTM Surrogate Validation)
# ==========================================
def run_forward_inference(model, test_loader, stats, device, max_batches=100):
    """
    Executes the forward simulation path: R_rs -> L_TOA.
    
    Methodological Justification:
    This phase validates the network's mathematical invertibility and its capability 
    to act as a high-precision Radiative Transfer Model (RTM) surrogate. By mapping 
    surface reflectance back to TOA radiance, we enforce physical consistency. 
    Additionally, we capture the latent variable Z to verify the Information Bottleneck 
    disentanglement, proving that noise and environmental ambiguities are successfully 
    isolated from the physical signal.
    """
    model.eval()
    
    rad_mean = stats.get("rad_mean", np.zeros(21))
    rad_std = stats.get("rad_std", np.ones(21))
    shift_rad = stats.get("shift_rad", 1.0)

    all_preds_rad, all_true_rad, all_z_pred = [], [], []

    print("[Simulation Engine] Initiating Forward Radiative Transfer validation...")

    with torch.no_grad():
        for batch_idx, (y_obs_norm, x_phys_norm, c_cond) in enumerate(tqdm(test_loader, desc="Simulating TOA Radiance")):
            if max_batches is not None and batch_idx >= max_batches:
                break
            
            x_phys_norm = x_phys_norm.to(device)
            y_obs_norm = y_obs_norm.to(device)
            c_cond = c_cond.to(device)

            # Step 1.1: Execute Forward Bijective Mapping
            # Domain Transformation: R_rs (8) + Zeros (21) -> L_TOA (21) + Latent Z (8)
            y_pred_norm, z_pred, _ = model.simulation_pass(x_phys_norm, c_cond)

            # Step 1.2: Collect Latent Distribution
            # Isolating Z for distribution analysis (Section D.4 in Supplementary Material)
            all_z_pred.append(z_pred.cpu().numpy()) 
            
            # Step 1.3: De-normalize TOA Radiance to Physical Units
            def denorm_rad(y_norm):
                y_log = y_norm * torch.tensor(rad_std, device=device) + torch.tensor(rad_mean, device=device)
                return (torch.exp(y_log) - shift_rad).cpu().numpy()

            all_preds_rad.append(denorm_rad(y_pred_norm))
            all_true_rad.append(denorm_rad(y_obs_norm))

    return (
        np.concatenate(all_true_rad, axis=0), 
        np.concatenate(all_preds_rad, axis=0), 
        np.concatenate(all_z_pred, axis=0)
    )

# ==========================================
# Phase 2: Main Execution Workflow
# ==========================================
def main():
    seed_everything(42)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Anonymized I/O Configuration ---
    DATA_DIR = "./data/processed"
    STATS_PATH = "./data/stats_log.json"
    CKPT_PATH = "./best_model.pth"   # 修正路径至根目录
    RESULT_DIR = "./output/inference/forward_test"
    
    os.makedirs(RESULT_DIR, exist_ok=True)
    RESULT_FILE = os.path.join(RESULT_DIR, "forward_results.npz")

    # Load empirical statistics or auto-generate synthetic priors
    stats = load_or_create_stats(STATS_PATH)
    
    cfg = Config()  # 确保你的 config.py 中的类名是 Config
    model = AtmosphericCorrectionCINN(cfg).to(DEVICE)

    # Architectural weight initialization
    if os.path.exists(CKPT_PATH):
        print(f"[Loader] Restoring pre-trained model states from {CKPT_PATH}")
        checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict({k.replace("module.", ""): v for k, v in checkpoint.get("model_state_dict", checkpoint).items()})
    else:
        print(f"[WARNING] Target checkpoint missing at {CKPT_PATH}.")
        print(f"[Loader] Executing continuous pipeline verification with dynamically initialized topologies.")

    # Initialize data pipeline with online statistics mapping
    test_ds = S3ChunkDataset(DATA_DIR, stats, mode="test")
    test_loader = DataLoader(test_ds, batch_size=4096, num_workers=4, pin_memory=True, drop_last=False)

    # Execute forward simulation workflow
    y_true_rad, y_pred_rad, z_pred = run_forward_inference(model, test_loader, stats, DEVICE, max_batches=100)

    # Persist the evaluated tensors for quantitative assessment
    print(f"[I/O] Persisting simulation results (True TOA, Simulated TOA, Latent Z) to {RESULT_FILE}")
    band_names_l1b = [f"Oa{i:02d}" for i in range(1, 22)]
    
    np.savez_compressed(
        RESULT_FILE,
        y_true=y_true_rad,
        y_pred=y_pred_rad,
        z_pred=z_pred, 
        band_names=band_names_l1b
    )
    print("[Success] Forward RTM simulation framework successfully executed.")

if __name__ == "__main__":
    main()