default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=100),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
log_level = 'INFO'
load_from = 'work_dirs/mocov3_resnet50_8xb512-amp-coslr-100e_in1k_transformers/epoch_100.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=2,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_sigmoid=True),
        topk=(
            1,
            2,
        )),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.1),
        dict(type='CutMix', alpha=1.0),
    ]))
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
bgr_mean = [
    103.53,
    116.28,
    123.675,
]
bgr_std = [
    57.375,
    57.12,
    58.395,
]
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
        hparams=dict(pad_val=[
            104,
            116,
            124,
        ], interpolation='bicubic')),
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
    dict(type='PackInputs'),
]
train_dataloader = dict(
    batch_size=32,
    num_workers=8,
    dataset=dict(
        type='CustomDataset',
        data_root='data/transformersv2',
        data_prefix='train',
        ann_file='meta/train.txt',
        pipeline=[
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
                    pad_val=[
                        104,
                        116,
                        124,
                    ], interpolation='bicubic')),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/transformersv2',
        data_prefix='val',
        ann_file='meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=236,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(
    type='Accuracy', topk=(
        1,
        2,
    ))
test_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/transformersv2',
        data_prefix='val',
        ann_file='meta/val.txt',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='ResizeEdge',
                scale=236,
                edge='short',
                backend='pillow',
                interpolation='bicubic'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(
    type='Accuracy', topk=(
        1,
        2,
    ))
optim_wrapper = dict(
    optimizer=dict(type='Lamb', lr=0.002, weight_decay=0.0001),
    paramwise_cfg=dict(bias_decay_mult=0.0, norm_decay_mult=0.0))
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=95,
        eta_min=1e-06,
        by_epoch=True,
        begin=5,
        end=100),
]
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=2048)

