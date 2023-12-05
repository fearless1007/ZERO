from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 64
LEARNING_RATE = 3e-3
#LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 120
WARMUP_EPOCHS = 3

# model config
IMG_SIZE = 32
PATCH_SIZE = 4
HIDDEN_SIZE = 2048
DEPTH = 12
NUM_HEADS = 16
MLP_RATIO = 4
NUM_CLASSES = 10
CHECKPOINT = False
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE)**2 + 1    # add 1 for cls token

# parallel setting
TENSOR_PARALLEL_SIZE = 8
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

#fp16 = dict(mode=AMP_TYPE.NAIVE)
#cliip_grad_norm = 1.0
clip_grad_norm = 1.0

#gradient_accumulation = 16

# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']