# learning notes
## 数据集
lightning利用L.LightningDataModule类作为数据集的整合抽象层，在其内部我们可以使用一个Dataset类，如果用自己的数据集也可以创建一个Dataset类的子类，只要其有getitem和len。
lightning会自动将数据转移到正确的设备，除非是自定义的数据类型。

## 超参数
利用LightningCLI和yaml可以管理超参数并自动注入。

## torch&torchvision
* torch
  * nn
    * 卷积  
    * Conv2d(in_ch, out_ch, kernel_size, stride=1, padding=0|("same"|"valid"), dilation=1, groups=1, bias=True, padding_mode='zeros'|'reflect'|'replicate'|'circular', *, device=None, dtype=None)
    * ConvTranspose2d(in_ch, out_ch, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', *, device=None, dtype=None)
    * 填充
    * ReflectionPad2d(padding: int | (left, right, top, bottom))
    * ReplicationPad2d / ZeroPad2d / ConstantPad2d：复制、零、常数填充，参数约定与上同。
    * functional
      * pad(input, pad=(...), mode='constant'|'reflect'|'replicate'|'circular', value=0)：N 维通用填充。
      * relu(x, inplace=False)
      * interpolate(input, size=None, scale_factor=None, mode=..., align_corners=None)（更灵活，推荐）
        * 说明：对线性族（bilinear/bicubic…），默认 align_corners=False；开启 True 会改变几何对齐方式，易造成数值差异。一般保留默认即可。
        * 与上面模块同名的函数基本一一对应（如 F.relu/conv2d/avg_pool2d/pad 等），适合自定义前向或不想持久化状态时用。
      * cross_entropy(...)
      * binary_cross_entropy_with_logits(...)
      * batch_norm（按batch统计，每通道）
      * instance_norm（按样本统计，每通道）
      * layer_norm（按最后若干特征维整体统计）
      * group_norm（把通道分组后，组内做统计）
      * rms_norm（用均方根而非方差）
      * local_response_norm（跨通道的局部响应归一化，AlexNet 时代遗产）
      * relu(x, inplace)
      * gelu(x, approx)
      * silu(x, inplace)
      * hardtanh(x,…) 
      * l1_loss（MAE）reduction={'none','mean','sum'}。更抗离群点
      * mse_loss（L2/平方误差）同样支持 reduction；梯度在误差大时更大。 
      * huber_loss（L1 与 L2 的折中）关键参：delta（阈值）；小误差用 L2，大误差转 L1，稳定又平滑。delta=1 近似 SmoothL1。
      * smooth_l1_loss（检测里常见，β=1 默认）关键参：beta。小于 β 用 L2，否则 L1。
      * cross_entropy（多类，输入=logits）参：weight（类权重）、ignore_index、label_smoothing、reduction。等价 LogSoftmax + NLLLoss。
      * binary_cross_entropy_with_logits（二分类/多标签）参：pos_weight（正例加权）、weight、reduction；把 Sigmoid 与 BCE 合成以提升数值稳定性。
    * 非线性
    * ReLU(inplace=False)：inplace=True 可降内存但会就地改写输入。
    * LeakyReLU(negative_slope=0.01, inplace=False)
    * PReLU(num_parameters=1, init=0.25)
    * SiLU()（=Swish）
    * GELU(approximate='none'|'tanh')
    * ReLU6(inplace=False)
    * 池化
    * MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    * AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
    * AdaptiveAvgPool2d(output_size: int | (h, w))：不管输入多大，最后都变成给定输出尺寸（常用于 GAP：output_size=1）
    * 归一化
    * BatchNorm2d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
    * InstanceNorm2d(num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False)
    * GroupNorm(num_groups, num_channels, eps=1e-5, affine=True)
    * LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True)
      * 选型套路：BN（按 batch/通道统计），IN（按样本/通道，风格化常用），GN（小 batch 稳定），LN（Transformer/序列常用）。
    * Dropout
    * Dropout(p=0.5, inplace=False)（全连接常用）
    * Dropout2d(p=0.5, inplace=False)（按通道丢弃；卷积特征图正则化）
    * 线性层
    * Linear(in_features, out_features, bias=True)；函数式：F.linear(x, W, b)
    * 上采样，插值
    * Upsample(size=None, scale_factor=None, mode='nearest'|'bilinear'|'bicubic'|..., align_corners=None)（模块化封装）
    * 损失函数
    * CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)（多类，输入为 logits；内部含 LogSoftmax+NLLLoss）
    * BCEWithLogitsLoss（二分类/多标签，输入=logits）将 Sigmoid 与 BCE 合并，数值更稳定；参：pos_weight（正例加权）、weight、reduction。优于“Sigmoid+BCELoss”的裸拼。
    * 
    * 激活函数
    * Tanh()  
    * ReLU(inplace=…) 
    * Sigmoid()  
    * GELU(approx=…) 
    * SiLU(inplace=…)
    * Hardtanh(min,max)
    * Tanhshrink()  
  * tanh(x) 
  * sigmoid(x) 
  * BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)（二分类/多标签，带 Sigmoid 的稳定实现）
  * optim
    * sgd(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=false, , maximize=false, foreach=none, differentiable=false, fused=none)
    * adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=false, , foreach=none, fused=none)
    * lr_scheduler
      * StepLR(optimizer, step_size, gamma=0.1) 每过 step_size 个 epoch，把 LR 乘以 gamma。适合简单的分段阶梯衰减。调用：每个 epoch 结束 scheduler.step()
      * MultiStepLR(optimizer, milestones, gamma=0.1) 在 milestones 指定的 epoch 点（如 [30, 60, 90]）把 LR 乘以 gamma。比 StepLR 更可控。
      * ExponentialLR(optimizer, gamma) 每个 epoch 让 LR 乘以 gamma（指数衰减）。
      * CosineAnnealingLR(optimizer, T_max, eta_min=0.0) 在 T_max 个 epoch 内从初始 LR 余弦下降到 eta_min。经典 cosine 退火。
      * CosineAnnealingWarmRestarts(optimizer, T_0, T_mult=1, eta_min=0.0) 余弦退火 + 周期性重启（SGDR）。每 T_i 个 epoch 重启到峰值；T_mult 控制每次周期是否变长。
      * 
  * utils
    * data
    * DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, persistent_workers=False, timeout=0, ...)   
      * 经验：num_workers>0 启多进程；GPU 训练常配 pin_memory=True；分布式用 sampler 控制切分。
  * AMP
  * autocast(device_type, dtype=None, enabled=True, cache_enabled=None)：前向与 loss 处混合精度区域。
  * amp
    * GradScaler(...)
  * cuda 
    * amp
      * GradScaler(...)：梯度缩放以避免 FP16 下溢（bf16 常不需要）。常见搭配：with autocast(): loss = ...; scaler.scale(loss).backward(); scaler.step(optim); scaler.update()。

* torchvision
  * transforms
    * Resize(size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None) size=int|(h, w)；给 int 时按短边等比缩放；可设 max_size 限制长边。
    * RandomResizedCrop(size, scale=(0.08,1.0), ratio=(3/4,4/3), interpolation=..., antialias=True)
    * Normalize(mean, std, inplace=False)：仅支持 Tensor（形状 [C,H,W] 或批处理）；逐通道 (x-mean)/std。
    * v2
      * Resize(...)：v2 版本，在自动处理 max_size 与 TVTensors 更完善。
  * wrap_dataset_for_transforms_v2(dataset)：令传统 datasets.* 产出 TVTensors 以兼容 v2。
  * datasets
    * ImageFolder(root, transform=None, target_transform=None, loader=None, is_valid_file=None) 目录结构：root/class_x/xxx.png；常与 transforms.* 组合。
  * io
    * read_image(path, mode=ImageReadMode.UNCHANGED, apply_exif_orientation=False)：读入 uint8 Tensor（RGB/灰度），形状 [C,H,W]。
  * ops
    * nms(boxes, scores, iou_threshold) -> idxs：标准 NMS；boxes 形如 [N,4]、格式 xyxy。
    * batched_nms(boxes, scores, idxs, iou_threshold)：按 idxs（通常类别 id）分组的 NMS。
    * roi_align(input[N,C,H,W], boxes, output_size, spatial_scale=1.0, sampling_ratio=-1, aligned=False)：RoI Align（Mask R-CNN 同款，默认平均池化）。

## LightningModule
在优化器方面其提供了使用多个优化器以及调度器的方法，对每个运行一遍流程，我们可以用optimizer_idx来区别。
新版本已经不支持这个了，其推荐手动更新。

## 配置整理
为了能够稳定的实现之前的三个需求，稳定的自动的项目下conda环境指定，在wsl中使用，以项目根目录为根能自由引用。同时应对目前存在的一些问题，无法稳定引用存在的文件，错误的引用旧文件。我使用新的架构，即将所有可导入代码放入src中，同时在一个文件夹下，此文件夹的名字为包的名字，而数据集，即使是软链接，或者文档等都在src外面。在外面我们通过pyproject.toml来配置，然后在wsl中打开相应环境用如下命令建立：
```bash
# 升级构建/安装工具（可选但推荐）
python -m pip install -U pip setuptools

# 可编辑安装（PEP 660）
pip install -e .
```
此后所有可导入文件可以通过软件包名下引入，而外部的应以项目根目录为根结合Path使用，如下：
```bash
from pathlib import Path
self.out_dir  = Path(out_dir).expanduser().resolve()
```

## LightningCLI
利用LightningCLI我们可以更科学的管理超参数，路径等等。
大部分我们需要的参数都已经在Lightning中规划好了位置，所以我们只要在yaml文件中按照格式一层层自CLI到Trainer等等，按照其缩进格式安排参数，就能够自动注入，然后就能方便，统一的管理参数。这是开始CLI的基本设置，给出CLI，并自动导入配置文件：
重点给出L.LightningDataModule和L.LightningModule的两个子类。
```python
from lightning.pytorch.cli import LightningCLI

def cli_main():
    cli = LightningCLI(parser_kwargs={"default_config_files": ["configs/cg.yaml"]})

if __name__ == "__main__":
    cli_main()
```

## 琐碎处理
绝大多数琐碎的工具都可以通过写在yaml中直接注入来调用很方便
```yaml
fit: 
  seed_everything: 42 # 所有lightning中的随机器或什么的种子，增强可复现性。

  #ckpt_path: "best"      # 等价于命令行 --ckpt_path=best
  #ckpt_path: logs/checkpoints/ # 需要恢复的记录
  trainer:
  ## test
    #fast_dev_run: 5 # 快速测试几个batch
    limit_train_batches: 0.25 # 每次训练随机取用数据集的比例，下面是验证集的
    limit_val_batches: 0.05
    #num_sanity_val_steps: 2 # 训练前测试

    profiler: # 关于时间的报告
      class_path: lightning.pytorch.profilers.SimpleProfiler
      init_args: { dirpath: logs/profilers, filename: perf_logs }
    default_root_dir: logs/checkpoints # 所有状态存储的地方
    max_epochs: 200 # 最大数量
    accelerator: gpu 
    devices: 1
    precision: 16-mixed # 调整精度使得能够更快
    log_every_n_steps: 50 # 每batch输入
    #deterministic: true # 确定性的，可复现，但会引发一些由于不确定导致的问题
    callbacks:
      - class_path: lightning.pytorch.callbacks.ModelCheckpoint # 监视，确定什么时候保存完整的checkpoint
        init_args: { monitor: val/Total_Loss, mode: min, save_top_k: 2, filename: "epoch{epoch}-valloss{val/loss:.3f}" }
      - class_path: lightning.pytorch.callbacks.EarlyStopping # 早停原则设置
        init_args: { monitor: val/Total_Loss, min_delta: 0.00, mode: min, patience: 10, check_finite: true }
      - class_path: lightning.pytorch.callbacks.ModelSummary # 模型报告的显示，多少层
        init_args: { max_depth: 2 }
      - class_path: lightning.pytorch.callbacks.DeviceStatsMonitor # GPU等的使用情况的监视
        init_args: {cpu_stats: true}
    logger:
      class_path: lightning.pytorch.loggers.TensorBoardLogger 
      init_args: { save_dir: logs, name: cg }

  model:
    # 多模型的给出，作为LightningModule的参数，还有其他参数
    G:
      class_path: cg.models.Generator.Generator
    F:
      class_path: cg.models.Generator.Generator
    Dx:
      class_path: cg.models.Discriminator.Discriminator
    Dy:
      class_path: cg.models.Discriminator.Discriminator

    # 2) 损失权重 & 训练超参
    lambda_cyc: 10.0     # (= 你的 l1)
    lambda_id: 5.0       # (= 你的 l2)
    n_epochs: 100
    n_epochs_decay: 100

      # 3) 多优化器（依赖注入）

      #opt_G:
      #  class_path: torch.optim.Adam
      #  init_args: { lr: 2.0e-4, betas: [0.5, 0.999] }
      #opt_D:
      #  class_path: torch.optim.Adam
      #  init_args: { lr: 2.0e-4, betas: [0.5, 0.999] }

      # 可选：调度器
      # sch_G:
      #   class_path: torch.optim.lr_scheduler.CosineAnnealingLR
      #   init_args: { T_max: 200, eta_min: 1.0e-7 }
      # sch_D: ...

  data:
    #class_path: cg.data.monet2photo.Monet2PhotoDM
    #init_args:
    data_dir: data/monet2photo 
    batch_size: 1
    size: 128
    resize: 143
    num_workers: 4
    seeds: 42
```



