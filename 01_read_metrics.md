
# Loading and Plotting A Single Learning Curve

Here is a simple example, showing how to load a single learning curve with
95% confidence range using `logger.read_metrics` call.

The plotting code is minimal to keep it simple.

Import the CommonMark X module
```python
from cmx import doc
from ml_logger import ML_Logger
```
Initialize the loader
```python
import os

loader = ML_Logger(root=os.getcwd(), prefix="data/walker-walk/curl")
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
avg, top, bottom, step = loader.read_metrics("train/episode_reward/mean@mean", "train/episode_reward/mean@84%",
                                             "train/episode_reward/mean@16%", x_key="step@mean",
                                             path="**/metrics.pkl", bin_size=40)
```
Step 2: Plot
```python
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
```

| **CURL on Walker-walk** |
|:-----------------------:|
| <img style="align-self:center; zoom:20%;" src="figures/learning_curve.png" image="None" styles="{'margin': '0.5em'}" width="None" height="None" dpi="300"/> |

## How Come The Figure Looks So Good?

This is because we place the following file: [./matplotlibrc](./matplotlibrc) inside
this folder. This file contains the following styling options:

```python

savefig.dpi: 226
savefig.bbox: tight
savefig.pad_inches: 0
savefig.transparent: False

axes.linewidth     : 2
axes.spines.top    : False
axes.spines.right  : False

axes.grid          : True    # display grid or not
axes.grid.axis     : both    # which axis the grid should apply to
axes.grid.which    : major # gridlines at {major, minor, both} ticks

# GRID
grid.color     : b0b0b0  # grid color
grid.linestyle : --      # solid
grid.linewidth : 0.8     # in points
grid.alpha     : 1.0     # transparency, between 0.0 and 1.0


font.family        : serif
font.serif         : Times New Roman
font.sans-serif    : Helvetica, Avant Garde, Computer Modern Sans Serif
font.cursive       : Zapf Chancery
font.monospace     : Courier, Computer Modern Typewriter
font.size           : 18.0
axes.titlesize      : 28.0
axes.titlepad       : 20
axes.labelsize      : large
axes.labelweight    : 300

lines.linewidth      : 2

mathtext.rm          : serif
mathtext.it          : serif:italic
mathtext.bf          : serif:bold
mathtext.fontset     : custom

legend.frameon       : False
legend.scatterpoints : 3
legend.fontsize      : medium
legend.markerscale   : 4
legend.handlelength  : 0.8
legend.handleheight  : 0
legend.handletextpad : 0.5

#axes.color_cycle    : b, g, r, c, m, y, k  # color cycle for plot lines

### TICKS
# see http://matplotlib.org/api/axis_api.html#matplotlib.axis.Tick
xtick.minor.visible  : True
ytick.minor.visible  : True
xtick.major.size     : 6      # major tick size in points
xtick.minor.size     : 4      # minor tick size in points
xtick.major.width    : 2      # major tick width in points
xtick.minor.width    : 1      # minor tick width in points
#xtick.labelsize     : large  # fontsize of the tick labels

ytick.major.size     : 6      # major tick size in points
ytick.minor.size     : 4      # minor tick size in points
ytick.major.width    : 2      # major tick width in points
ytick.minor.width    : 1      # minor tick width in points
#ytick.labelsize     : large  # fontsize of the tick labels

