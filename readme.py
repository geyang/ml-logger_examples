import matplotlib.pyplot as plt
from cmx import doc
from ml_logger import ML_Logger, BinOptions

doc @ """
# Loading and Plotting Learning Curves

"""

PREFIX = "geyang/dmc_gen/01_baselines/train/walker-walk/curl"

with doc @ """Initialize the loader""":
    loader = ML_Logger("http://improbable005.csail.mit.edu:8080", prefix=PREFIX)

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    doc.print(files)

with doc @ """Step 1: load the data""":
    step, avg, top, bottom = loader.read_metrics("step", "train/episode_reward/mean", "train/episode_reward/mean@95%",
                                                 "train/episode_reward/mean@5%", path="**/metrics.pkl")

with doc @ "Step 2: Plot", doc.table().figure_row() as r:
    title = "CURL"

    plt.figure(figsize=(3, 2))

    plt.plot(step.to_list(), avg.to_list())
    plt.fill_between(step, bottom, top, alpha=0.15)

    r.savefig(f"figures/learning_curve.png", title="Learning Curve", dpi=300)

doc.flush()
