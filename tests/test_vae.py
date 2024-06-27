import torch
from opensora.registry import MODELS, build_module
import torchvision
from opensora.datasets.utils import get_transforms_image, get_transforms_video
from opensora.datasets import save_sample
import copy

vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="stabilityai/sd-vae-ft-ema",
    micro_batch_size=128,
) 

image_size=(144, 256)
transform_name="resize_crop"
transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

if __name__ == "__main__":

    vae = build_module(vae, MODELS)
    vae = vae.to('cuda').eval()
    video_path = "/slurmhome/kzhang/datasets/HD-VG-130M/0I6vQFn93dI_scene-068.mp4"
    vframes, _, _ = torchvision.io.read_video(filename=video_path, pts_unit="sec", output_format="TCHW")
    print(vframes.shape)
    print(type(vframes))
  
    transform = transforms["video"]
    video = transform(vframes)  # T C H W
    video = video.permute(1, 0, 2, 3)  # C T B H W
    video = video.unsqueeze(0)  # 1 C T B H W
    original_video = copy.deepcopy(video)
    save_sample(original_video[0], fps=30, save_path='original')
    video = video.to('cuda')
   
    print(video.shape)
    print(type(video))
    with torch.no_grad():
        video = vae.encode(video)  # [B, C, T, H, W]
        print(video.shape)
        samples = vae.decode(video)
        print(samples.shape)
    
    save_sample(samples[0], fps=30, save_path='output')