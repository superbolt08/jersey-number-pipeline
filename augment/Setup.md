# Setup

Add the augment_data.py functions to:
str/parseq/strhub/data/augment.py

## Registering the Functions

### Name the functions:

under `_RAND_TRANSFORMS.extend` add:
'Oclussion',
'ColorJitter',
'ElasticDistortion',
'PerspectiveWarp'

### Set the hyperparams:

under `auto_augment.LEVEL_TO_ARG.update` add:
'Oclussion':partial(\_level_to_arg, max=50),
'ColorJitter': partial(\_level_to_arg, max=10),
'PerspectiveWarp': partial(\_level_to_arg, max=20),
'ElasticDistortion': partial(\_level_to_arg, max=8)

### Map the function to the name:

under `auto_augment.NAME_TO_OP.update` add:
'Oclussion':occlusion,
'ColorJitter':color_jitter,
'PerspectiveWarp': perspective_warp,
'ElasticDistortion': elastic_distortion,
