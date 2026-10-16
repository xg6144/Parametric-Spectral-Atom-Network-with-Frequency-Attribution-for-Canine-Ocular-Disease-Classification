import os
from typing import List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset

class CanineOcularDataset(Dataset):
    def __init__(self, root_dir: str, split: str, img_size: int = 224, transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.img_size = img_size
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx = {}

        if os.path.exists(self.root_dir):
            candidates = sorted([d.name for d in os.scandir(self.root_dir) if d.is_dir()])
            self.classes = candidates
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            print(f"[{split.upper()}] Found {len(self.classes)} classes: {self.classes}")

            for cls_name in self.classes:
                cls_idx = self.class_to_idx[cls_name]
                d_path = os.path.join(self.root_dir, cls_name)
                
                for fn in os.listdir(d_path):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(d_path, fn), cls_idx))

        print(f"[{split.upper()}] Loaded {len(self.samples)} images (Multiclass).")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f'Image decode failed: {path}') from exc

        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(y, dtype=torch.long)
