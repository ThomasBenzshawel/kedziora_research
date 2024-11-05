import argparse
import torch

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(args):
    out = args.image_file.split(args.sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        # print(image_file)
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        try:
            image = load_image(image_file)
            out.append(image)
        except Exception as e:
            print(f"Error loading image {image_file}: {e}")
    return out


def eval_model(args, obj, model, model_name, tokenizer, image_processor):
    # Model


    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    image_files = image_parser(args)
    images = load_images(image_files)
    image_sizes = [x.size for x in images]
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .cuda()
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(outputs)

    # save the output to a text file in object_descriptions folder
    with open(f"/home/benzshawelt/Research/object_descriptions/{obj}.txt", "w") as f:
        f.write(outputs)


model_path = "liuhaotian/llava-v1.5-7b"

prompt = """Y# Improved LLaVA Prompt for 3D Model Description

You are an expert 3D model classifier with access to front, back, right, left, up, and down perspectives of a model. Your task is to describe the model in intricate detail, focusing on the following aspects:

1. Overall shape and structure
2. Color palette and textures
3. Distinctive features or unique elements
4. Proportions and scale (if discernible)
5. Any text or markings visible on the object
6. Functional elements or moving parts (if applicable)

Your description should be detailed enough for someone who has never seen the model to accurately recreate it using a language-based 3D model generator.

Here's an example of a good response:

Example Object: Coffee Mug

Description:
The object is a cylindrical coffee mug with a rounded bottom and a slightly tapered top. Its overall height is approximately 4 inches, with a diameter of 3 inches at the base, narrowing to 2.8 inches at the rim.

The mug's primary color is a deep, matte navy blue that covers the entire exterior surface. The interior of the mug is glazed in a contrasting off-white color, visible when viewing from the top perspective.

A distinctive feature of this mug is its handle, which attaches to the right side of the cylinder. The handle is curved in a 'C' shape, with a smooth, ergonomic design that allows for a comfortable grip. It's made of the same material as the mug body and shares the navy blue color.

The mug's surface texture appears smooth and uniform, with a slight sheen visible under direct light, suggesting a ceramic or porcelain material. There are no visible patterns or decorations on the exterior.

On the bottom of the mug, there's a small, circular indentation about 2 inches in diameter. This forms a stable base for the mug and likely bears the manufacturer's mark, though it's not visible in the provided images.

The rim of the mug is smooth and slightly rounded, without any chips or irregularities. There's a subtle lip around the top edge, which helps prevent spills and adds to the mug's aesthetic appeal.

No text or additional markings are visible on any side of the mug.

This description covers the mug's shape, color, distinctive features, proportions, material properties, and functional elements, providing a comprehensive overview for accurate 3D model generation.

Now, please describe the object in the provided images with similar attention to detail."""

# Time this loop

import time

start = time.time()

import os

#directory walk over "/home/benzshawelt/Research/objaverse_images/"

objects = os.listdir("/home/benzshawelt/Research/objaverse_images")


disable_torch_init()



model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)

for obj in objects:
    image_files_array = os.listdir("/home/benzshawelt/Research/objaverse_images/" + obj)

    image_files = ""
    for image_file in image_files_array:
        image_files += "/home/benzshawelt/Research/objaverse_images/" + obj + "/" + image_file + ","
    
    if image_files == "":
        continue
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": image_files,
        "sep": ",",
        "temperature": 0.6,
        "top_p": 0.9,
        "num_beams": 1,
        "max_new_tokens": 8000
    })()

    eval_model(args, obj, model, model_name, tokenizer, image_processor)


end = time.time()

print(f"Time taken: {end - start} seconds")
print(f"Time taken: {(end - start) / 60} minutes")