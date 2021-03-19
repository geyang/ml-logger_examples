
# Loading and Plotting Learning Curves


Initialize the loader
```python
loader = ML_Logger("http://improbable005.csail.mit.edu:8080", prefix=PREFIX)
```
Check all the files
```python
files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
doc.print(files)
```

```
['400/metrics.pkl', '100/metrics.pkl', '300/metrics.pkl', '200/metrics.pkl']
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
