from cmx import doc

doc @ """
# Loading and Plotting A Single Learning Curve

Here is a simple example, showing how to load a single learning curve with
95% confidence range using `logger.read_metrics` call.

The plotting code is minimal to keep it simple.
"""
with doc @ """Import the CommonMark X module""":
    from cmx import doc
    from ml_logger import ML_Logger

with doc @ """Initialize the loader""":
    import os

    loader = ML_Logger(root=os.getcwd(), prefix="data/walker-walk/curl")

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    doc.print(files)

with doc @ """Step 1: load the data""":
    avg, top, bottom, step = loader.read_metrics("train/episode_reward/mean@mean", "train/episode_reward/mean@84%",
                                                 "train/episode_reward/mean@16%", x_key="step@mean",
                                                 path="**/metrics.pkl", bin_size=40)

with doc @ "Step 2: Plot", doc.table().figure_row() as r:
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    title = "CURL on Walker-walk"

    plt.figure()

    plt.plot(step, avg.to_list(), color="#23aaff")
    plt.fill_between(step, bottom, top, color="#23aaff", alpha=0.15)

    plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
    plt.title(title)
    plt.xlabel("Steps")
    plt.ylabel("Return")

    r.savefig(f"figures/learning_curve.png", title=title, dpi=300, zoom="20%")

doc @ """
## How Come The Figure Looks So Good?

This is because we place the following file: [./matplotlibrc](./matplotlibrc) inside
this folder. This file contains the following styling options:

```python
"""
doc @ loader.load_text("../../../matplotlibrc")
"""
```
"""

doc.flush()