

_base_ = [
    '../../_base_/models/deform_DETR_r50_16xb2_50e_coco.py',
    '../../_base_/datasets/full_mdmt_deform.py', '../../_base_/default_runtime.py'
]

# img_scale = (800, 1440)
img_scale = (1333, 800)
samples_per_gpu = 4

model = dict(
    type='ByteTrack',
    detector = dict(
        type='DeformableDETR',
        backbone=dict(
            type='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
        neck=dict(
            type='ChannelMapper',
            in_channels=[512, 1024, 2048],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
        bbox_head=dict(
            type='DeformableDETRHead',
            num_query=300,
            num_classes=4,
            in_channels=2048,
            sync_cls_avg_factor=True,
            # as_two_stage=True,
            as_two_stage=False,
            transformer=dict(
                type='DeformableDetrTransformer',
                encoder=dict(
                    type='DetrTransformerEncoder',
                    num_layers=6,
                    transformerlayers=dict(
                        type='BaseTransformerLayer',
                        attn_cfgs=dict(
                            type='MultiScaleDeformableAttention', embed_dims=256),
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
                decoder=dict(
                    type='DeformableDetrTransformerDecoder',
                    num_layers=6,
                    return_intermediate=True,
                    transformerlayers=dict(
                        type='DetrTransformerDecoderLayer',
                        attn_cfgs=[
                            dict(
                                type='MultiheadAttention',
                                embed_dims=256,
                                num_heads=8,
                                dropout=0.1),
                            dict(
                                type='MultiScaleDeformableAttention',
                                embed_dims=256)
                        ],
                        feedforward_channels=1024,
                        ffn_dropout=0.1,
                        operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                        'ffn', 'norm')))),
            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True,
                offset=-0.5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=2.0),
            # with_box_refine=True,
            with_box_refine=False,
            loss_bbox=dict(type='L1Loss', loss_weight=5.0),
            loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
        test_cfg=dict(max_per_img=100),

        # pretrained='checkpoints/epoch_12.pth',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            # 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth'  # noqa: E501
            #    './checkpoint/faster_rcnn_r50_fpn_carafe_1x_full_mdmt/mmdet_4_train/epoch_12.pth'  # noqa: E501
            #  '../mydev/home/wjy/checkpoints/deformable-detr-refine-twostage_r50_16xb2-50e_coco/deform_detr_3/epoch_45.pth'
             '../mydev/home/wjy/deform_mmdet2_28/deformable_detr_r50_16x2_50e_coco/epoch_50.pth'
        )
        ),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.7, tentative=0.5),
        num_frames_retain=30),
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset_type = 'MOTChallengeDataset'
data_root = '/home/wjy/wujingyi/Dataset/MOT17challenge/'

backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[{
            'type':
            'RandomChoiceResize',
            'scales': [(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                       (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                       (736, 1333), (768, 1333), (800, 1333)],
            'keep_ratio':
            True
        }],
                    [{
                        'type': 'RandomChoiceResize',
                        'scales': [(400, 4200), (500, 4200), (600, 4200)],
                        'keep_ratio': True
                    }, {
                        'type': 'RandomCrop',
                        'crop_type': 'absolute_range',
                        'crop_size': (384, 600),
                        'allow_negative_crop': True
                    }, {
                        'type':
                        'RandomChoiceResize',
                        'scales': [(480, 1333), (512, 1333), (544, 1333),
                                   (576, 1333), (608, 1333), (640, 1333),
                                   (672, 1333), (704, 1333), (736, 1333),
                                   (768, 1333), (800, 1333)],
                        'keep_ratio':
                        True
                    }]]),
    dict(type='PackDetInputs')
]
# test_pipeline = [
#     dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
#     dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(
#         type='PackDetInputs',
#         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                    'scale_factor'))
# ]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        visibility_thr=-1,
        ann_file=data_root + 'annotations/train_cocoformat.json',
        img_prefix=data_root + 'train',
        ref_img_sampler=dict(
            num_ref_imgs=1,
            frame_range=10,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_cocoformat.json',
        img_prefix=data_root + 'val',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/val_cocoformat.json',
        # ann_file=data_root + 'annotations/half-train_cocoformat.json',
        img_prefix=data_root + 'val',
        ref_img_sampler=None,
        pipeline=test_pipeline))





# # use d img_norm
# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))







# img_norm_cfg = dict(
#     mean=[102.9801, 115.9465, 122.7717], std=[1.0, 1.0, 1.0], to_rgb=False)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', with_bbox=True),
#     dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size_divisor=32),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=8,
#     workers_per_gpu=4,
#     train=dict(pipeline=train_pipeline),
#     val=dict(pipeline=test_pipeline),
#     test=dict(pipeline=test_pipeline))
optimizer = dict(lr=0.002, paramwise_cfg=dict(norm_decay_mult=0.))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=5000,
    warmup_ratio=1.0 / 5000,
    step=[40, 55])
total_epochs = 60
runner = dict(type='EpochBasedRunner', max_epochs=60)

# optimizer_config = dict(grad_clip=None)

# some hyper parameters
# num_last_epochs = 10
# resume_from = None
# interval = 5



# custom_hooks = [
#     dict(
#         type='YOLOXModeSwitchHook',
#         num_last_epochs=num_last_epochs,
#         priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=num_last_epochs,
#         interval=interval,
#         priority=48),
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0001,
#         priority=49)
# ]

checkpoint_config = dict(interval=1)
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512.))













































