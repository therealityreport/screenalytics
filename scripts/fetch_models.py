import os
from insightface.app import FaceAnalysis

root = os.path.expanduser(os.getenv("INSIGHTFACE_HOME", "~/.insightface"))
pack = os.getenv("INSIGHTFACE_PACK", "buffalo_l")
app = FaceAnalysis(name=pack, root=root, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Models ready at:", os.path.join(root, "models", pack))
