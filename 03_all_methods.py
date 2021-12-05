from cmx import doc

doc @ """
# Facet and Grouping

Here we show the learning curve from multiple methods, on the same domain.

We typically arrange the data with a folder structure `f"{domain}/{method}/{seed}"`.
"""

with doc @ """Initialize the loader""":
    import os

    from cmx import doc
    import matplotlib.pyplot as plt
    from matplotlib import ticker
    from ml_logger import ML_Logger

    loader = ML_Logger(root=os.getcwd(), prefix="data/walker-walk")

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    doc.print(files)

with doc @ """Plotting A Single Time Series""":
    def group(xKey="step", yKey="train/episode_reward/mean", color=None, bins=50, label=None, dropna=False):
        avg, top, bottom, step = loader.read_metrics(f"{yKey}@mean", f"{yKey}@84%", f"{yKey}@16%", x_key=f"{xKey}@mean",
                                                     path="**/metrics.pkl", num_bins=bins, dropna=dropna)
        plt.plot(step, avg, color=color, label=label)
        plt.fill_between(step, bottom, top, alpha=0.15, color=color)

with doc @ "Step 2: Plot", doc.table().figure_row() as r:
    colors = ['#49b8ff', '#444444', '#ff7575', '#66c56c', '#f4b247']

    for method in ['curl', 'rad', 'pad']:
        plt.figure()

        with loader.Prefix(method):
            group(yKey="episode_reward/mean", bins=None, dropna=True, color=colors[0], label="Eval")
            group(yKey="train/episode_reward/mean", color=colors[1], label="Train")
            plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k" if x else "0"))
            plt.legend(frameon=False)
            plt.ylim(0, 1000)

        r.savefig(f"figures/{method}/train_vs_eval.png", title=method.capitalize(), dpi=300)

doc.flush()
