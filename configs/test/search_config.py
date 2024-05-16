# this file is only for batch size search and is not used for training

# Define dataset
dataset = dict(
    type="VariableVideoTextDataset",
    data_path=None,
    num_frames=None,
    frame_interval=3,
    image_size=(None, None),
    transform_name="resize_crop",
)

# bucket config format:
# 1. { resolution: {num_frames: (prob, batch_size)} }, in this case batch_size is ignored when searching
# 2. { resolution: {num_frames: (prob, (max_batch_size, ))} }, batch_size is searched in the range [batch_size_start, max_batch_size), batch_size_start is configured via CLI
# 3. { resolution: {num_frames: (prob, (min_batch_size, max_batch_size))} }, batch_size is searched in the range [min_batch_size, max_batch_size)
# 4. { resolution: {num_frames: (prob, (min_batch_size, max_batch_size, step_size))} }, batch_size is searched in the range [min_batch_size, max_batch_size) with step_size (grid search)
# 5. { resolution: {num_frames: (0.0, None)} }, this bucket will not be used

bucket_config = {
    "144p": {40: (1.0, (8, 48, 8)), 50: (1.0, (8, 36, 8))},
    "256": {20: (1.0, (8, 56, 8)), 30: (1.0, (8, 32, 8)), 40: (1.0, (8, 24, 8)), 50: (1.0, (8, 18, 8))},
}


# Define acceleration
num_workers = 16
num_bucket_build_workers = 16
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model
model = dict(
    type="STDiT2-XL/2",
    from_pretrained=None,
    input_sq_size=512,  # pretrained model is trained on 512x512
    qk_norm=True,
    enable_flashattn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=4,
    local_files_only=True,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=200,
    shardformer=True,
    local_files_only=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 1000
log_every = 10
ckpt_every = 1000
load = None

batch_size = None
lr = 2e-5
grad_clip = 1.0
