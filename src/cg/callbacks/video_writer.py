import cv2, numpy as np, os
from lightning.pytorch.callbacks import BasePredictionWriter
import torch

class VideoWriterCallback(BasePredictionWriter):
    def __init__(self, out_path, fps, size, codec="mp4v", is_color=True):
        super().__init__(write_interval="batch")
        self.out_path, self.fps, self.size = out_path, float(fps), tuple(size)  # (W,H)
        self.codec, self.is_color = codec, is_color
        self._writer = None

    def setup(self, trainer, pl_module, stage):
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, self.size, self.is_color)
        if not self._writer.isOpened():
            raise RuntimeError(f"open VideoWriter failed: {self.out_path}")

    def write_on_batch_end(self, trainer, pl_module, prediction, *args, **kwargs):
        if not trainer.is_global_zero: return
        x = prediction
        if isinstance(x, torch.Tensor):
            if torch.is_floating_point(x):
                x = (x.clamp(-1,1)+1)*127.5
            x = x.to(torch.uint8)
            if x.dim() == 3: x = x.unsqueeze(0)
            if x.dim() == 4 and x.shape[1] in (1,3):
                x = x.permute(0,2,3,1)                 # [N,H,W,C]
            frames = x.cpu().numpy()
        else:
            frames = np.array(x)

        Wt,Ht = self.size
        for f in frames:
            # 现在 predict_step 已保证原 H,W；这里不再 resize，只转 BGR
            self._writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    def on_predict_epoch_end(self, trainer, pl_module):
        if self._writer is not None:
            self._writer.release(); self._writer = None




