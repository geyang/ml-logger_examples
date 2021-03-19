from cmx import doc
from ml_logger import ML_Logger

doc @ """
# Download Data from Server

You can directly read from the server, which is the standard use-flow. We
download the data here so that you can process the data locally.
"""

PREFIX = "geyang/dmc_gen/01_baselines/train"

with doc @ """Initialize the loader""":
    loader = ML_Logger("http://<some-dash-server>:8080", prefix=PREFIX)

with doc @ """Check all the files""":
    files = loader.glob(query="**/metrics.pkl", wd=".", recursive=True)
    for path in files:
        loader.download_file(path, to="data/", relative=True)
    doc.print(*files, sep="\n")

doc.flush()
