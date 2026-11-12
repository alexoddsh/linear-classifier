from pathlib import Path
import modal
from .classifier import main

ROOT_PATH = Path(__file__).parent
VOLUME_PATH = "/volumes"

app = modal.App("linear-class", include_source=False)
image = (
        modal.Image.debian_slim()
        .env({"USE_GPU": "1"})
        .pip_install("numpy", "cupy-cuda12x[ctk]")
        .add_local_python_source("linear_classifier")
        .add_local_dir("src/linear_classifier/data", remote_path="/root/linear_classifier/data")
    )

volume = modal.Volume.from_name("linear-class")

@app.cls(image=image, volumes={"/volume": volume}, gpu=["A10"], timeout=60 * 60 * 4)
class RemoteRun:
    @modal.method()
    def class_runner(self, args):
        import shutil
        main(args)
        shutil.copy(f"{ROOT_PATH / "data/output.npy"}", "/volume/output/output.npy")
        volume.commit()
