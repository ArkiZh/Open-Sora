active_steps = 3
base_frames = 40
base_resolution = '144p'
batch_size = None
batch_size_end = 48
batch_size_start = 8
batch_size_step = 8
bucket_config = dict({
    '144p': {20: (1.0, 56), 30: (1.0, 40), 40: (1.0, 32), 50: (1.0, 24)},
    '256': {20: (1.0, 32), 30: (1.0, 24), 40: (1.0, 16), 50: (1.0, 16)}
})
ckpt_every = 1000
config = 'configs/test/search_config.py'
dataset = dict(
    data_path='/slurmhome/kzhang/datasets/HD-VG-130M/data.csv',
    frame_interval=3,
    image_size=(
        None,
        None,
    ),
    num_frames=None,
    transform_name='resize_crop',
    type='VariableVideoTextDataset')
dtype = 'bf16'
epochs = 1000
grad_checkpoint = True
grad_clip = 1.0
load = None
log_every = 10
lr = 2e-05
mask_ratios = None
model = dict(
    enable_flashattn=True,
    enable_layernorm_kernel=True,
    from_pretrained=None,
    input_sq_size=512,
    qk_norm=True,
    type='STDiT2-XL/2')
multi_resolution = False
num_bucket_build_workers = 16
num_workers = 16
output = 'configs/test/search_result-node2.py'
outputs = 'outputs'
plugin = 'zero2'
scheduler = dict(timestep_respacing='', type='iddpm')
seed = 42
sp_size = 1
start_from_scratch = False
text_encoder = dict(
    from_pretrained='DeepFloyd/t5-v1_1-xxl',
    local_files_only=True,
    model_max_length=200,
    shardformer=True,
    type='t5')
vae = dict(
    from_pretrained='stabilityai/sd-vae-ft-ema',
    local_files_only=True,
    micro_batch_size=4,
    type='VideoAutoencoderKL')
wandb = False
warmup_steps = 2
