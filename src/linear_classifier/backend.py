import os
if os.environ.get("USE_GPU") == "1":
    import cupy as xp
else:
    import numpy as xp
