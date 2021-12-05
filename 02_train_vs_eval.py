import os

import matplotlib.pyplot as plt
from cmx import doc
from matplotlib import ticker

from ml_logger import ML_Logger

doc @ """
# Comparing Two Learning Curves Side-by-side

Here we compare the training performance versus the performance 
on the evaluation domain.

We show the training performance in gray, to accentuate the 
evaluation curve.
"""

with doc @ """Initialize the loader""":
    loader = ML_Logger(root=os.getcwd(), prefix="data/walker-walk/curl")

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    doc.print(files)

with doc @ """A Single Time Series""":
    def group(xKey="step", yKey="train/episode_reward/mean", color=None, bin=10, label=None, dropna=False):
        avg, top, bottom, step = loader.read_metrics(f"{yKey}@mean", f"{yKey}@84%", f"{yKey}@16%", x_key=f"{xKey}@mean",
                                                     path="**/metrics.pkl", bin_size=bin, dropna=dropna)
        plt.plot(step, avg, color=color, label=label)
        plt.fill_between(step, bottom, top, alpha=0.15, color=color)
        return avg

with doc @ "Step 2: Plot", doc.table().figure_row() as r:
    colors = ['#49b8ff', '#444444', '#ff7575', '#66c56c', '#f4b247']

    avg = group(yKey="episode_reward/mean", bin=None, color=colors[0], label="Eval")
    group(yKey="train/episode_reward/mean", color=colors[1], label="Train")

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
    plt.legend()
    plt.title("Walker-walk")
    plt.xlabel("Steps")
    plt.ylabel("Return")

    r.savefig(f"figures/train_vs_eval.png", title="Train VS Eval", dpi=300, zoom="20%")
    plt.close()

doc @ """
## Where does the empty cuts come from? 

These cuts are places where the `avg` is `NaN`. You can just filter this out 
in the `group` function.
"""
with doc:
    doc.print(avg)

doc @ """
## How to fix this?

You can turn on the `dropna` flag, which is OFF by default.
"""
with doc, doc.table().figure_row() as r:

    avg = group(yKey="episode_reward/mean", bin=None, color=colors[0], label="Eval", dropna=True)

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
    plt.legend()
    plt.title("Walker-walk")
    plt.xlabel("Steps")
    plt.ylabel("Return")

    r.savefig(f"figures/train_vs_eval_dropna.png", title="Train VS Eval", dpi=300, zoom="20%")

doc.flush()
