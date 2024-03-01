import torch
from PIL import Image
import numpy as np
import hat.archs
import hat.data
import hat.models
from basicsr.models import build_model
from basicsr.utils import get_root_logger
from basicsr.utils.options import dict2str, parse_options
from os import path as osp



def simple_test(root_path,image_path, output_path):
    # Initialize logger
    logger = get_root_logger()

    # Configuration options (Placeholder, replace with your actual config)
    opt, _ = parse_options(root_path, is_train=False)

    # Load model
    model = build_model(opt)
    model=model.get_bare_model()
    model.eval()
    torch_input = torch.randn(1, 3, 64, 64)
    onnx_program = torch.onnx.export(model, torch_input,r"D:\HAT\onnx_models\hat_test.onnx")

    # # Load image
    # img = Image.open(image_path).convert('RGB')
    # img = np.array(img)
    # img = torch.from_numpy(img).float().unsqueeze(0).permute(0, 3, 1, 2) / 255.0  # Adjust normalization as needed
    #
    # # Inference
    # with torch.no_grad():
    #     output = model(img)
    #
    # # Save image (assumes output is a torch tensor in the range [0, 1])
    # output_img = output.squeeze().permute(1, 2, 0).cpu().numpy() * 255.0
    # output_img = output_img.astype(np.uint8)
    # Image.fromarray(output_img).save(output_path)
    #
    # logger.info(f"Output saved to {output_path}")


if __name__ == '__main__':
    root_path = r"D:\HAT\options\test\test_HAT_SRx4_DS_dataset_from_scratch.yml"
    output_path = r'D:\HAT\test_Image\OST_009_crop_LR_out.png'
    image_path=r"D:\HAT\test_Image\OST_009_crop_LR.png"
    simple_test(root_path, image_path, output_path)
