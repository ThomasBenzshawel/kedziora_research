import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import sys
import warnings
import os
import tqdm


def image_data_augmentation(folder_path):
    # Image paths for local files

    image_paths =[]
    for image in os.listdir(folder_path):
        image_paths.append(os.path.join(folder_path, image))

    

    images = []
    images = [Image.open(image_path) for image_path in image_paths]

    image_tensors = process_images(images, image_processor, model.config)
    image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in image_tensors]

    # Prepare interleaved text-image input
    conv_template = "qwen_1_5"

    question = f"""You are an expert 3D model describer, have 6 images of the same object
    {DEFAULT_IMAGE_TOKEN} This is the first image. Can you describe what you see?\n\n
    Now, let's look at another image of the same object from a different view: {DEFAULT_IMAGE_TOKEN}\n
    Can you describe what you see in this image?\n\n
    Now, let's look at the other images of the same object from a different views: {DEFAULT_IMAGE_TOKEN}\n
    {DEFAULT_IMAGE_TOKEN}\n
    What do you see in this image?\n\n
    {DEFAULT_IMAGE_TOKEN}\n
    What do you see in this image?\n\n
    {DEFAULT_IMAGE_TOKEN}\n
    What do you see in this image?\n\n
    Please describe what the object that you have been shown in as much detail as possible.
    Such that it can be used to create a 3D model of the object using generative AI.
    Never mention the word "3D model" in your description, or explain what you are thinking.
    Please be concise yet extremely descriptive in your response. 
    First classify, then provide a one liner label for the object. Then provide a detailed description of the object.
    An example of your output could be: 

    "Label: Cat
    Short Description: A brown cat with a pink nose and green eyes, sitting on a white windowsill looking out the window.
    Long Description: A small brown cat with a pink nose and green eyes, sitting on a white windowsill looking out the window. 
    The cat has a fluffy tail and is looking out the window with a curious expression. The window has a white frame. 
    The cat has a small pink nose and green eyes. The cat is sitting in a relaxed position with its tail curled around its body. 
    The cat has a fluffy coat and is looking out the window with a curious expression."
    """

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size for image in images]

    # Generate response
    cont = model.generate(
        input_ids,
        images=image_tensors,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    # print(text_outputs[0])

    # Save output to txt file with the same name as the folder
    # Grab the name of the folder from the path
    folder_name = folder_path.split("/")[-1]

    #output file goes to objaverse_descriptions folder
    os.makedirs("/home/benzshawelt/Research/objaverse_descriptions", exist_ok=True)
    output_file = f"/home/benzshawelt/Research/objaverse_descriptions/{folder_name}.txt"
    # Write output to file, if it already exists, overwrite it, if not create a new one
    with open(output_file, "w") as f:
        f.write(text_outputs[0])
    print(f"Output saved to {output_file}")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Model configuration
    pretrained = "lmms-lab/llava-onevision-qwen2-7b-ov"
    model_name = "llava_qwen"
    device = "cuda:0"

    model_name = "llava_qwen"
    device_map = "cuda:0"
    llava_model_args = {
            "multimodal": True,
            # "attn_implementation": None,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
    model.eval()
    model.to(device)

    target_folder = "/home/benzshawelt/Research/objaverse_images"

    # Use tqdm to show progress bar

    for folder in tqdm.tqdm(os.listdir(target_folder)):
        folder_path = os.path.join(target_folder, folder)
        # if the folder is empty, skip it
        if not os.listdir(folder_path):
            continue
        image_data_augmentation(folder_path)
