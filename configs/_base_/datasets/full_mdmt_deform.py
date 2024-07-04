# dataset settings

# train_dataloader = dict(
#     batch_size=4,
#     num_workers=2,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=True),
#     batch_sampler=dict(type='AspectRatioBatchSampler'),
#     dataset=dict(
#         type='MdmtDataset',
#         data_root='../Dataset/MOT17challenge/',
#         ann_file='annotations/train_cocoformat.json',
#         data_prefix=dict(img='train/'),
#         filter_cfg=dict(filter_empty_gt=False, min_size=32),
#         pipeline=[
#             dict(
#                 type='LoadImageFromFile',
#                 file_client_args=dict(backend='disk')),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(type='RandomFlip', prob=0.5),
#             dict(
#                 type='RandomChoice',
#                 transforms=[[{
#                     'type':
#                     'RandomChoiceResize',
#                     'scales': [(480, 1333), (512, 1333), (544, 1333),
#                                (576, 1333), (608, 1333), (640, 1333),
#                                (672, 1333), (704, 1333), (736, 1333),
#                                (768, 1333), (800, 1333)],
#                     'keep_ratio':
#                     True
#                 }],
#                             [{
#                                 'type': 'RandomChoiceResize',
#                                 'scales': [(400, 4200), (500, 4200),
#                                            (600, 4200)],
#                                 'keep_ratio': True
#                             }, {
#                                 'type': 'RandomCrop',
#                                 'crop_type': 'absolute_range',
#                                 'crop_size': (384, 600),
#                                 'allow_negative_crop': True
#                             }, {
#                                 'type':
#                                 'RandomChoiceResize',
#                                 'scales':
#                                 [(480, 1333), (512, 1333), (544, 1333),
#                                  (576, 1333), (608, 1333), (640, 1333),
#                                  (672, 1333), (704, 1333), (736, 1333),
#                                  (768, 1333), (800, 1333)],
#                                 'keep_ratio':
#                                 True
#                             }]]),
#             dict(type='PackDetInputs')
#         ]))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='MdmtDataset',
#         data_root='../Dataset/MOT17challenge/',
#         ann_file='annotations/val_cocoformat.json',
#         data_prefix=dict(img='val/'),
#         test_mode=True,
#         pipeline=[
#             dict(
#                 type='LoadImageFromFile',
#                 file_client_args=dict(backend='disk')),
#             dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='PackDetInputs',
#                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                            'scale_factor'))
#         ]))
# val_evaluator = dict(
#     type='CocoMetric',
#     ann_file='../Dataset/MOT17challenge/annotations/val_cocoformat.json',
#     metric='bbox',
#     format_only=False)
# test_dataloader = val_dataloader
# test_evaluator = val_evaluator


# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='MdmtDataset',
#         data_root='../Dataset/MOT17challenge/',
#         ann_file='annotations/test_cocoformat.json',
#         data_prefix=dict(img='test/'),
#         test_mode=True,
#         pipeline=[
#             dict(
#                 type='LoadImageFromFile',
#                 file_client_args=dict(backend='disk')),
#             dict(type='Resize', scale=(1333, 800), keep_ratio=True),
#             dict(type='LoadAnnotations', with_bbox=True),
#             dict(
#                 type='PackDetInputs',
#                 meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
#                            'scale_factor'))
#         ]))
# test_evaluator = dict(
#     type='CocoMetric',
#     ann_file='../Dataset/MOT17challenge/annotations/test_cocoformat.json',
#     metric='bbox',
#     format_only=False)



