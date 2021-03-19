
# Download Data from Server

You can directly read from the server, which is the standard use-flow. We
download the data here so that you can process the data locally.

Initialize the loader
```python
loader = ML_Logger("http://<some-dash-server>:8080", prefix=PREFIX)
```
Check all the files
```python
files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
for path in files:
    loader.download_file(path, to="data/", relative=True)
doc.print(*files, sep="\n")
```

```
finger-spin/pad/100/metrics.pkl
finger-spin/pad/200/metrics.pkl
finger-spin/rad/200/metrics.pkl
finger-spin/soda/200/metrics.pkl
finger-spin/curl/200/metrics.pkl
ball_in_cup-catch/pad/400/metrics.pkl
ball_in_cup-catch/pad/100/metrics.pkl
ball_in_cup-catch/pad/300/metrics.pkl
ball_in_cup-catch/pad/200/metrics.pkl
ball_in_cup-catch/rad/400/metrics.pkl
ball_in_cup-catch/rad/100/metrics.pkl
ball_in_cup-catch/rad/300/metrics.pkl
ball_in_cup-catch/rad/200/metrics.pkl
ball_in_cup-catch/soda/400/metrics.pkl
ball_in_cup-catch/soda/100/metrics.pkl
ball_in_cup-catch/soda/300/metrics.pkl
ball_in_cup-catch/soda/200/metrics.pkl
ball_in_cup-catch/curl/400/metrics.pkl
ball_in_cup-catch/curl/100/metrics.pkl
ball_in_cup-catch/curl/300/metrics.pkl
ball_in_cup-catch/curl/200/metrics.pkl
cartpole-swingup/pad/400/metrics.pkl
cartpole-swingup/pad/100/metrics.pkl
cartpole-swingup/pad/300/metrics.pkl
cartpole-swingup/pad/200/metrics.pkl
cartpole-swingup/rad/400/metrics.pkl
cartpole-swingup/rad/100/metrics.pkl
cartpole-swingup/rad/300/metrics.pkl
cartpole-swingup/rad/200/metrics.pkl
cartpole-swingup/soda/400/metrics.pkl
cartpole-swingup/soda/100/metrics.pkl
cartpole-swingup/soda/300/metrics.pkl
cartpole-swingup/soda/200/metrics.pkl
cartpole-swingup/curl/400/metrics.pkl
cartpole-swingup/curl/100/metrics.pkl
cartpole-swingup/curl/300/metrics.pkl
cartpole-swingup/curl/200/metrics.pkl
walker-walk/pad/400/metrics.pkl
walker-walk/pad/100/metrics.pkl
walker-walk/pad/300/metrics.pkl
walker-walk/pad/200/metrics.pkl
walker-walk/rad/400/metrics.pkl
walker-walk/rad/100/metrics.pkl
walker-walk/rad/300/metrics.pkl
walker-walk/rad/200/metrics.pkl
walker-walk/soda/400/metrics.pkl
walker-walk/soda/100/metrics.pkl
walker-walk/soda/300/metrics.pkl
walker-walk/soda/200/metrics.pkl
walker-walk/curl/400/metrics.pkl
walker-walk/curl/100/metrics.pkl
walker-walk/curl/300/metrics.pkl
walker-walk/curl/200/metrics.pkl
```
