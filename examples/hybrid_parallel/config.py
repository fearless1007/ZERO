from colossalai.amp import AMP_TYPE

# hyperparameters
# BATCH_SIZE is as per GPU
# global batch size = BATCH_SIZE x data parallel size
BATCH_SIZE = 32
LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.3
NUM_EPOCHS = 1
WARMUP_EPOCHS = 1

# model config
IMG_SIZE = 224
PATCH_SIZE = 16
HIDDEN_SIZE = 1024
DEPTH = 12
NUM_HEADS = 16
MLP_RATIO = 4
NUM_CLASSES = 1000
CHECKPOINT = False
SEQ_LENGTH = (IMG_SIZE // PATCH_SIZE)**2 + 1    # add 1 for cls token

# parallel setting
TENSOR_PARALLEL_SIZE = 4
TENSOR_PARALLEL_MODE = '1d'

parallel = dict(
    pipeline=1,
    tensor=dict(mode=TENSOR_PARALLEL_MODE, size=TENSOR_PARALLEL_SIZE),
)

fp16 = dict(mode=AMP_TYPE.NAIVE)
cliip_grad_norm = 1.0
#clip_grad_norm = 1.0

#gradient_accumulation = 16

# pipeline config
NUM_MICRO_BATCHES = parallel['pipeline']
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LENGTH, HIDDEN_SIZE)