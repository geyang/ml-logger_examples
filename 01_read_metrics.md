
# Loading and Plotting A Single Learning Curve

Here is a simple example, showing how to load a single learning curve with
95% confidence range using `logger.read_metrics` call.

The plotting code is minimal to keep it simple.

Initialize the loader
```python
loader = ML_Logger(prefix="data/walker-walk/curl")
```
Check all the files
```python
files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
doc.print(files)
```

```
['300/metrics.pkl', '400/metrics.pkl', '100/metrics.pkl', '200/metrics.pkl']
```
Step 1: load the data
```python
step, avg, top, bottom = loader.read_metrics("step",
                                             "train/episode_reward/mean",
                                             "train/episode_reward/mean@95%",
                                             "train/episode_reward/mean@5%",
                                             path="**/metrics.pkl",
                                             bin=BinOptions(key="step", size=40))
```
Step 2: Plot
```python
title = "CURL"

plt.figure(figsize=(3, 2))

plt.plot(step.to_list(), avg.to_list())
plt.fill_between(step, bottom, top, alpha=0.15)

r.savefig(f"figures/learning_curve.png?ts={doc.now()}", title="Learning Curve", dpi=300)
```

| **Learning Curve** |
|:------------------:|
| <img style="align-self:center;" src="figures/learning_curve.png?ts=2021-03-19 17:12:32.493298-04:00" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |
