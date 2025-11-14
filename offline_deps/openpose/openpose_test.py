from open_pose import OpenposeDetector
import torch
from PIL import Image, ImageOps, ImageSequence
import numpy as np

model = OpenposeDetector.from_pretrained().to("cuda")


def load_image(image_path):

    img = Image.open(image_path)

    output_images = []
    output_masks = []
    w, h = None, None

    excluded_formats = ['MPO']

    for i in ImageSequence.Iterator(img):
        i = ImageOps.exif_transpose(i)

        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")

        if len(output_images) == 0:
            w = image.size[0]
            h = image.size[1]

        if image.size[0] != w or image.size[1] != h:
            continue

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        elif i.mode == 'P' and 'transparency' in i.info:
            mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        output_images.append(image)
        output_masks.append(mask.unsqueeze(0))

    if len(output_images) > 1 and img.format not in excluded_formats:
        output_image = torch.cat(output_images, dim=0)
        output_mask = torch.cat(output_masks, dim=0)
    else:
        output_image = output_images[0]
        output_mask = output_masks[0]

    return (output_image, output_mask)

def save_image(img, output_path):
    pil_image = Image.fromarray(img)
    pil_image.save(output_path)


if __name__ == '__main__':

    image, _ = load_image(r"test01.webp")
    image = np.asarray(image * 255., dtype=np.uint8)
    pose_img, openpose_dict = model(image[0], include_body=True, include_face=False, include_hand=False,
                                    output_type="np",
                                    detect_resolution=512, image_and_json=True, xinsr_stick_scaling=False)

    print(pose_img.shape)
    save_image(pose_img,"out.png")

    if 'people' in openpose_dict and openpose_dict["people"]:
        print(openpose_dict)



