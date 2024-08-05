_base_ = [
    'resnet50_8xb256-rsb-a3-100e_transformers.py',
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

test_dataloader = dict(
    batch_size=256,
    num_workers=5,
    dataset=dict(
        type='CustomDataset',
        data_root='data/infer-transformers',
        data_prefix='test',
        # ann_file='meta/test.txt',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_evaluator = dict(type='Predict', topk=(1, 2))
