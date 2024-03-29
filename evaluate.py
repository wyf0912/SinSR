import pyiqa
import os
import argparse
from pathlib import Path
import torch
from utils import util_image
import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def evaluate(in_path, ref_path, ntest):
    metric_dict = {}
    metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    metric_paired_dict = {}
    
    in_path = Path(in_path) if not isinstance(in_path, Path) else in_path
    assert in_path.is_dir()
    
    ref_path_list = None
    if ref_path is not None:
        ref_path = Path(ref_path) if not isinstance(ref_path, Path) else ref_path
        ref_path_list = sorted([x for x in ref_path.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: ref_path_list = ref_path_list[:ntest]
        
        metric_paired_dict["psnr"]=pyiqa.create_metric('psnr', test_y_channel=True, color_space='ycbcr').to(device)
        metric_paired_dict["lpips"]=pyiqa.create_metric('lpips').to(device)
        metric_paired_dict["ssim"]=pyiqa.create_metric('ssim', test_y_channel=True, color_space='ycbcr' ).to(device)
        
    lr_path_list = sorted([x for x in in_path.glob("*.[jpJP][pnPN]*[gG]")])
    if ntest is not None: lr_path_list = lr_path_list[:ntest]
    
    print(f'Find {len(lr_path_list)} images in {in_path}')
    result = {}
    for i in tqdm.tqdm(range(len(lr_path_list))):
        in_path = lr_path_list[i]
        ref_path = ref_path_list[i] if ref_path_list is not None else None
        
        im_in = util_image.imread(in_path, chn='rgb', dtype='float32')  # h x w x c
        im_in_tensor = util_image.img2tensor(im_in).cuda()              # 1 x c x h x w
        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                result[key] = result.get(key, 0) + metric(im_in_tensor).item()
        
        if ref_path is not None:
            im_ref = util_image.imread(ref_path, chn='rgb', dtype='float32')  # h x w x c
            im_ref_tensor = util_image.img2tensor(im_ref).cuda()    
            for key, metric in metric_paired_dict.items():
                result[key] = result.get(key, 0) + metric(im_in_tensor, im_ref_tensor).item()
                
    for key, res in result.items():
        print(f"{key}: {res/len(lr_path_list):.5f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-r", "--ref_path", type=str, default=None)
    parser.add_argument("--ntest", type=int, default=None)
    args = parser.parse_args()
    evaluate(args.in_path, args.ref_path, args.ntest)
    


