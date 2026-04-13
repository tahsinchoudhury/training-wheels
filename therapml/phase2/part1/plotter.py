from pathlib import Path
import matplotlib.pyplot as plt

class LossPlotter:
    def plot_losses(self, *, train_losses: list[float], eval_losses: list[float], out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

        max_len = max(len(train_losses), len(eval_losses))

        # Pad shorter list with NaN so both have same length
        padded_train = train_losses + [float('nan')] * (max_len - len(train_losses))
        padded_eval = eval_losses + [float('nan')] * (max_len - len(eval_losses))

        intervals = list(range(1, max_len + 1))
        plt.figure()
        plt.plot(intervals, padded_train, label="train loss", color="tab:blue")
        plt.plot(intervals, padded_eval, label="eval loss", color="tab:orange")
        plt.xlabel("interval")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()