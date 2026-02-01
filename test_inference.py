import numpy as np

from inference import load_cnn, load_vit, predict_move_cnn, predict_move_vit


def make_random_board(seed=0, fill_prob=0.35):
    rng = np.random.default_rng(seed)
    board = np.zeros((6, 7), dtype=np.int8)

    # Fill columns from bottom up (roughly realistic boards)
    for c in range(7):
        h = 0
        while h < 6 and rng.random() < fill_prob:
            r = 5 - h
            board[r, c] = 1 if rng.random() < 0.5 else -1
            h += 1

    return board


def test_model(model_type="cnn", n_tests=5):
    if model_type == "cnn":
        model, device, ckpt = load_cnn(device="cpu")
    else:
        model, device, ckpt = load_vit(device="cpu")

    print(f"\n=== Testing {model_type.upper()} ===")
    print("Architecture:", ckpt.get("arch"))
    print("Best val_acc:", ckpt.get("best_val"))

    for i in range(n_tests):
        board = make_random_board(seed=i + 10)

        move = (
            predict_move_cnn(model, device, board, player=1)
            if model_type == "cnn"
            else predict_move_vit(model, device, board, player=1)
        )

        print(f"Test {i+1}")
        print(board)
        print("Predicted move:", move)
        print("-" * 30)


if __name__ == "__main__":
    test_model("cnn", n_tests=3)
    test_model("vit", n_tests=3)
