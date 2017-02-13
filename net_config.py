'''
file with various network architectures.
'''

NET_CONFIGS = {
    'openai': {
        'conv_filters': (32, 32, 32, 32),
        'conv_sizes': (3, 3, 3, 3),
        'conv_strides': (2, 2, 2, 2),
        'pads': ['same']*4,
        'conv_droputs': [0.0]*4,
        'fc_sizes': (256,),
        'fc_dropouts': [0.],
        'batch_norm': True,
        'activation': 'elu'
    },
    'openai2': {
        'conv_filters': (32, 32, 32, 32),
        'conv_sizes': (6, 6, 6, 6),
        'conv_strides': (2, 2, 2, 2),
        'pads': ['same'] * 4,
        'conv_droputs': [0.0] * 4,
        'fc_sizes': (256,),
        'fc_dropouts': [0.],
        'batch_norm': True,
        'activation': 'elu'
    },
    'dqn': {
        'conv_filters': (32, 64, 64),
        'conv_sizes': (8, 4, 3),
        'conv_strides': (4, 2, 1),
        'pads': ['valid']*3,
        'conv_droputs': [0.]*3,
        'fc_sizes': (512, 256),
        'fc_dropouts': [0.]*2,
        'batch_norm': True,
        'activation': 'relu'
    },
    'dqn2': {
        'conv_filters': (16, 32),
        'conv_sizes': (8, 4),
        'conv_strides': (4, 2),
        'pads': ['valid'] * 2,
        'conv_droputs': [0.0] * 2,
        'fc_sizes': (256, ),
        'fc_dropouts': [0.,],
        'batch_norm': True,
        'activation': 'relu'
    }
}
