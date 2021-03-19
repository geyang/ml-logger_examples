
# 4. Results Over All Domains




Initialize the loader
```python
loader = ML_Logger(prefix="data")
```
Check all the files
```python
files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
```
Plotting A Single Time Series
```python
def group(xKey="step", yKey="train/episode_reward/mean", color=None, bins=40, label=None):
    if bins:
        bins = BinOptions(key=xKey, size=bins)
    step, avg, top, bottom = loader.read_metrics(xKey, yKey, yKey + "@95%", yKey + "@5%",
                                                 path="**/metrics.pkl", bin=bins)
    plt.plot(step.to_list(), avg.to_list(), color=color, label=label)
    plt.fill_between(step, bottom, top, alpha=0.15, color=color)
```
Step 2: Plot
```python
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

            r.savefig(f"figures/{name}/{method}/train_vs_eval.png",
                      title=method.capitalize(), dpi=300)
```

Walker [walk]
| **Curl** | **Rad** | **Pad** | **Soda** |
|:--------:|:-------:|:-------:|:--------:|
| <img style="align-self:center;" src="figures/walker/curl/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/walker/rad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/walker/pad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/walker/soda/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |

Cartpole [swingup]
| **Curl** | **Rad** | **Pad** | **Soda** |
|:--------:|:-------:|:-------:|:--------:|
| <img style="align-self:center;" src="figures/cartpole/curl/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/cartpole/rad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/cartpole/pad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/cartpole/soda/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |

Ball In Cup [catch]
| **Curl** | **Rad** | **Pad** | **Soda** |
|:--------:|:-------:|:-------:|:--------:|
| <img style="align-self:center;" src="figures/ball_in_cup/curl/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/ball_in_cup/rad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/ball_in_cup/pad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/ball_in_cup/soda/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |

Finger [spin]
| **Curl** | **Rad** | **Pad** | **Soda** |
|:--------:|:-------:|:-------:|:--------:|
| <img style="align-self:center;" src="figures/finger/curl/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/finger/rad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/finger/pad/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> | <img style="align-self:center;" src="figures/finger/soda/train_vs_eval.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
