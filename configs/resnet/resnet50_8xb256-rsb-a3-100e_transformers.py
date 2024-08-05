_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/resnet50.py',
]

# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=6,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=236,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='data/transformersv2',
        data_prefix='train',
        ann_file='meta/train.txt',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/transformersv2',
        data_prefix='val',
        ann_file='meta/val.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 2))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Lamb', lr=0.002, weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=100)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)


# model settings
model = dict(
    backbone=dict(norm_cfg=dict(type='SyncBN', requires_grad=True)),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0)
    ]),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        # loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        loss=dict(use_sigmoid=True),
        topk=(1, 2),
    )
)

# evaluation = dict(interval=1, metric='accuracy', metric_options={'topk': (1, 2)})
test_evaluator = dict(type='Accuracy', topk=(1, 2))
val_evaluator = dict(type='Accuracy', topk=(1, 2))

# # schedule settings
# optim_wrapper = dict(
#     optimizer=dict(lr=0.002, momentum=0.9, weight_decay=0.0001),
#     paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
# )
default_hooks = dict(checkpoint=dict(max_keep_ckpts=3))

load_from = "work_dirs/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_transformers/epoch_300.pth"
