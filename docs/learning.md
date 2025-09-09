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
        * CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)（多类，输入为 logits；内部含 LogSoftmax+NLLLoss）
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