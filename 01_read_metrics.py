import matplotlib.pyplot as plt
from cmx import doc
from ml_logger import ML_Logger, BinOptions

doc @ """
# 1. Loading and Plotting A Single Learning Curve

Here is a simple example, showing how to load a single learning curve with
95% confidence range using `logger.read_metrics` call.

The plotting code is minimal to keep it simple.
"""

with doc @ """Initialize the loader""":
    loader = ML_Logger(prefix="data/walker-walk/curl")

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    doc.print(files)

with doc @ """Step 1: load the data""":
    step, avg, top, bottom = loader.read_metrics("step",
                                                 "train/episode_reward/mean",
                                                 "train/episode_reward/mean@95%",
                                                 "train/episode_reward/mean@5%",
                                                 path="**/metrics.pkl",
                                                 bin=BinOptions(key="step", size=40))

with doc @ "Step 2: Plot", doc.table().figure_row() as r:
    title = "CURL"

    plt.figure(figsize=(3, 2))

    plt.plot(step.to_list(), avg.to_list())
    plt.fill_between(step, bottom, top, alpha=0.15)

    r.savefig(f"figures/learning_curve.png", title="Learning Curve", dpi=300)

doc.flush()
