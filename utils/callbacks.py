import torch
import torch.distributed as dist

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, rank=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.rank = rank
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'max':
            if val_score < self.best_score + self.min_delta:
                self.counter += 1
                if self.rank == 0:
                    print(f"[EarlyStopping] {self.counter}/{self.patience} validation score did not improve.")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0
        else:
            if val_score > self.best_score - self.min_delta:
                self.counter += 1
                if self.rank == 0:
                    print(f"[EarlyStopping] {self.counter}/{self.patience} validation score did not improve.")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_score
                self.counter = 0

        if dist.is_initialized():
            flag_tensor = torch.tensor(int(self.early_stop), dtype=torch.int, device=torch.device('cuda', self.rank))
            dist.all_reduce(flag_tensor, op=dist.ReduceOp.MAX)
            self.early_stop = bool(flag_tensor.item())

        if self.early_stop and self.rank == 0:
            print("[EarlyStopping] Triggered! Stopping training.")
