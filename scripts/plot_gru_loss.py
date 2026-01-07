import matplotlib.pyplot as plt
from pathlib import Path

# Output path
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

# Epochs
epochs = list(range(1, 31))

# Loss values
train_loss = [
    0.6774, 0.6735, 0.6730, 0.6706, 0.6679,
    0.6656, 0.6613, 0.6528, 0.6280, 0.6030,
    0.5819, 0.5554, 0.5303, 0.4716, 0.4082,
    0.3805, 0.3669, 0.3557, 0.3428, 0.3377,
    0.3311, 0.3300, 0.3235, 0.3204, 0.3138,
    0.3116, 0.3093, 0.3052, 0.3020, 0.2967
]

val_loss = [
    0.6692, 0.6748, 0.6734, 0.6727, 0.6687,
    0.6658, 0.6615, 0.6472, 0.6177, 0.5899,
    0.5636, 0.5408, 0.5211, 0.4494, 0.4289,
    0.3942, 0.3806, 0.3735, 0.3613, 0.3562,
    0.3686, 0.3435, 0.3657, 0.3339, 0.3416,
    0.3248, 0.3455, 0.3397, 0.3298, 0.3428
]

# Plot
plt.figure(plt.figure(figsize=(6, 4)))
plt.plot(epochs, train_loss, label="Train loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training and Validation Loss")
plt.legend()
plt.grid(False)
plt.tight_layout()

# Save + show
out_path = OUT_DIR / "gru_loss_curves.png"
plt.savefig(out_path, dpi=150)
plt.show()

print(f"Plot gemt i: {out_path.resolve()}")
