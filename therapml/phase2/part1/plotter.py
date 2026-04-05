from pathlib import Path
import matplotlib.pyplot as plt

class LossPlotter:
    def plot_losses(self, *, train_losses: list[float], eval_losses: list[float], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = list(range(1, len(train_losses) + 1))
        plt.figure()
        plt.plot(epochs, train_losses, label="train loss", color="tab:blue")
        plt.plot(epochs, eval_losses, label="eval loss", color="tab:orange")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()