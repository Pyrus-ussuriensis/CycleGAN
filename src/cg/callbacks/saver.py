from lightning.pytorch import Callback
from torchvision.utils import make_grid, save_image
from pathlib import Path
import torch

class SampleSaver(Callback):
    def __init__(self, every_n_steps=500, out_dir="results/pictures", use_logger=True, value_range=(-1,1)):
        self.every_n_steps = every_n_steps
        self.out_dir = Path(out_dir)
        self.use_logger = use_logger
        self.value_range = value_range

    @torch.inference_mode()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gs = trainer.global_step
        ge = trainer.current_epoch
        if gs == 0 or gs % self.every_n_steps != 0:
            return

        m, p = batch  # Monet, Photo
        m, p = m.to(pl_module.device), p.to(pl_module.device)
        fake_m, fake_p = pl_module.G(p), pl_module.F(m)
        # 拼四宫格：真p -> 假m | 真m -> 假p
        grid = make_grid(
            [p[0], fake_m[0], m[0], fake_p[0]],
            nrow=4, normalize=True, value_range=self.value_range
        )

        if trainer.is_global_zero and self.out_dir:
            save_image(grid, f"{self.out_dir}/epoch={ge:04d}-step={gs:08d}.png")

        # 记录到日志器
        if self.use_logger and trainer.logger is not None:
            logger = trainer.logger
            if hasattr(logger, "experiment"):
                exp = logger.experiment
                # TensorBoard
                if hasattr(exp, "add_image"):
                    exp.add_image("samples/grid", grid, global_step=gs)
                # W&B（也支持原生 log_image）
                if hasattr(logger, "log_image"):
                    logger.log_image(key="samples", images=[grid.permute(1,2,0).cpu().numpy()])
