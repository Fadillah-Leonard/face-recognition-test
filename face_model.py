from typing import Optional
import numpy as np

class FaceModel:
    """
    Adapter ArcFace.

    Versi ini memakai InsightFace FaceAnalysis dengan model pretrained
    (buffalo_l). Tidak ada training.

    Kalau lu sudah punya script/model ArcFace sendiri, bagian file ini
    yang diganti. server.py dan database.py tidak perlu diubah.
    """

    def __init__(self):
        self.app = None

    def load(self):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.app is None:
            raise RuntimeError("Model belum di-load.")

        faces = self.app.get(image)

        # Beta sederhana: harus tepat satu wajah.
        if len(faces) != 1:
            return None

        embedding = np.asarray(faces[0].embedding, dtype=np.float32)

        # Normalize supaya cosine similarity bisa dihitung sebagai dot product.
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        embedding = embedding / norm
        return embedding
