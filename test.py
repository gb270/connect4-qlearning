import numpy as np
import os

def save_test_checkpoint(filename='checkpoint_test.npy'):
    test_q_table = {
        (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0): np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    }
    checkpoint = {
        'q_table': test_q_table,
        'epoch': 10,
        'epsilon': 0.1
    }
    np.save(filename, checkpoint)

def load_test_checkpoint(filename='checkpoint_test.npy'):
    if os.path.exists(filename):
        checkpoint = np.load(filename, allow_pickle=True).item()
        return checkpoint
    else:
        raise FileNotFoundError("Checkpoint file not found.")

if __name__ == "__main__":
    save_test_checkpoint()
    checkpoint = load_test_checkpoint()
    print("Loaded checkpoint:", checkpoint)
