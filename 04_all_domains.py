import matplotlib.pyplot as plt
from cmx import doc
from ml_logger import ML_Logger, BinOptions

doc @ """
# Results Over All Domains



"""

with doc @ """Initialize the loader""":
    loader = ML_Logger(prefix="data")

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)

with doc @ """Plotting A Single Time Series""":
    def group(xKey="step", yKey="train/episode_reward/mean", color=None, bins=40, label=None):
        if bins:
            bins = BinOptions(key=xKey, size=bins)
        step, avg, top, bottom = loader.read_metrics(xKey, yKey, yKey + "@95%", yKey + "@5%",
                                                     path="**/metrics.pkl", bin=bins)
        plt.plot(step.to_list(), avg.to_list(), color=color, label=label)
        plt.fill_between(step, bottom, top, alpha=0.15, color=color)

with doc @ "Step 2: Plot":
    title = "CURL"
    colors = ['#49b8ff', '#444444', '#ff7575', '#66c56c', '#f4b247']

    for domain in ['walker-walk', 'cartpole-swingup', 'ball_in_cup-catch', 'finger-spin']:
        name, task = domain.split("-")

        doc(name.replace('_', ' ').title(), f"[{task}]")
        with loader.Prefix(domain), doc.table().figure_row() as r:
            for method in ['curl', 'rad', 'pad', 'soda']:
                plt.figure(figsize=(3, 2))

                with loader.Prefix(method):
                    group(yKey="episode_reward/mean", color=colors[0], bins=None, label="Eval")
                    group(yKey="train/episode_reward/mean", color=colors[1], label="Train")
                    plt.legend(frameon=False)
                    plt.ylim(0, 1000)

                r.savefig(f"figures/{name}/{method}/train_vs_eval.png?ts={doc.now()}",
                          title=method.capitalize(), dpi=300)

doc.flush()
