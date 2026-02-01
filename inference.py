import os
import numpy as np
import torch

from app.cnn_model import Connect4StrongCNN
from app.vit_model import BoardViTStrong


# -----------------------------------------------------
# Paths
# -----------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(ROOT, "weights")

CNN_PATH = os.path.join(WEIGHTS_DIR, "best_cnn_2ch.pt")
VIT_PATH = os.path.join(WEIGHTS_DIR, "best_vit_2ch.pt")  # rename if yours differs


# -----------------------------------------------------
# Encoding: board (6,7) -> tensor (1,2,6,7)
# -----------------------------------------------------
def encode_board_2ch_tensor(board_6x7: np.ndarray, player: int, device):
    """
    board_6x7: np.array (6,7) with {-1,0,1}
    player: +1 or -1
    Returns x: torch.FloatTensor (1,2,6,7) from +1 POV
    """
    board = board_6x7 if player == 1 else (-board_6x7)

    plus = (board == 1).astype(np.float32)
    minus = (board == -1).astype(np.float32)

    x = np.stack([plus, minus], axis=0)  # (2,6,7)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)  # (1,2,6,7)
    return x


def legal_moves_mask(board_6x7: np.ndarray) -> np.ndarray:
    """
    True where legal.
    Column is illegal if top cell is non-zero.
    """
    return (board_6x7[0, :] == 0)


def pick_best_legal_move(logits_7: np.ndarray, legal_mask_7: np.ndarray) -> int:
    logits = logits_7.copy()
    logits[~legal_mask_7] = -1e9
    return int(np.argmax(logits))


# -----------------------------------------------------
# Loaders (use ckpt['model_config'] + ckpt['state_dict'])
# -----------------------------------------------------
def load_cnn(device: str | None = None, weights_path: str = CNN_PATH):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ckpt = torch.load(weights_path, map_location=device_t)
    config = ckpt["model_config"]  # {'channels':192,'n_blocks':12,'drop_path_max':0.1,'head_dropout':0.15}
    model = Connect4StrongCNN(**config).to(device_t)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device_t, ckpt


def load_vit(device: str | None = None, weights_path: str = VIT_PATH):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)

    ckpt = torch.load(weights_path, map_location=device_t)
    config = ckpt["model_config"]  # {'d_model':256,'nhead':8,'num_layers':8,'dim_ff':1024,'dropout':0.1}
    model = BoardViTStrong(**config).to(device_t)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device_t, ckpt


# -----------------------------------------------------
# Prediction APIs (AWS backend calls these)
# -----------------------------------------------------
@torch.no_grad()
def predict_move_cnn(model, device, board_6x7, player: int) -> int:
    board = np.array(board_6x7, dtype=np.int8)
    x = encode_board_2ch_tensor(board, player, device)          # (1,2,6,7)
    logits = model(x).squeeze(0).detach().cpu().numpy()         # (7,)
    legal = legal_moves_mask(board)
    return pick_best_legal_move(logits, legal)


@torch.no_grad()
def predict_move_vit(model, device, board_6x7, player: int) -> int:
    board = np.array(board_6x7, dtype=np.int8)
    x = encode_board_2ch_tensor(board, player, device)          # (1,2,6,7)
    logits = model(x).squeeze(0).detach().cpu().numpy()         # (7,)
    legal = legal_moves_mask(board)
    return pick_best_legal_move(logits, legal)


# -----------------------------------------------------
# Optional: unified entry point
# -----------------------------------------------------
def load_policy(which: str = "cnn", device: str | None = None):
    which = which.lower()
    if which == "cnn":
        return load_cnn(device=device)
    elif which in ("vit", "transformer"):
        return load_vit(device=device)
    else:
        raise ValueError("which must be 'cnn' or 'vit'/'transformer'")
