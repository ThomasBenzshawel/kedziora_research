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
    image_paths = []
    for image in os.listdir(folder_path):
        image_paths.append(os.path.join(folder_path, image))

    # Load all images
    images = [Image.open(image_path) for image_path in image_paths]

    if len(images) != 6:
        # skip this folder if it doesn't contain exactly 6 images
        print(f"Skipping folder {folder_path} as it does not contain exactly 6 images.")
        return
    
    # Process all images for later use
    all_image_tensors = process_images(images, image_processor, model.config)
    all_image_tensors = [_image.to(dtype=torch.float16, device=device) for _image in all_image_tensors]
    
    # Process images in pairs and build chain of thought
    descriptions = []
    conv_template = "qwen_1_5"
    
    # First pair of images with one-shot example
    pair1_question = f"""You are an expert object analyst with exceptional attention to detail. I'll show you multiple images of the same object from different angles.

    {DEFAULT_IMAGE_TOKEN} This is the first image of the object. 
    {DEFAULT_IMAGE_TOKEN} This is the second image of the same object from a different angle.
    
    Analyze these two images and describe only the physical object shown, completely ignoring backgrounds or supporting surfaces. I need precise details about shape, materials, colors, textures, and construction.
    
    Here's an example of an excellent analysis for a coffee mug:
    
    "This object is a ceramic coffee mug with a cylindrical body and a C-shaped handle attached to one side. The mug has a glossy white exterior with a blue geometric pattern consisting of interlocking triangles that form a band around the upper portion. From the second angle, I can see the interior is solid blue with the same glossy finish as the exterior. The mug has straight sides with a slight taper toward the base, and a flat bottom with an unglazed ring where it sits on surfaces. The rim is smooth and rounded, approximately 3.5 inches in diameter, while the handle has enough clearance for an adult's fingers. The construction appears sturdy, with the handle securely attached to the body at both the upper and lower connection points."
    
    Please provide a similarly detailed analysis for the object in these images. Focus exclusively on the physical characteristics of the object itself.
    """
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], pair1_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [images[0].size, images[1].size]
    
    # Generate response for first pair
    pair1_response = model.generate(
        input_ids,
        images=all_image_tensors[:2],
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )
    pair1_output = tokenizer.batch_decode(pair1_response, skip_special_tokens=True)[0]
    descriptions.append(pair1_output)
    
    # Second pair of images with context from first pair
    pair2_question = f"""Continuing your analysis of the same object.
    
    Your observations so far: 
    {pair1_output}
    
    Now examining two more angles of the same object:
    {DEFAULT_IMAGE_TOKEN} 
    {DEFAULT_IMAGE_TOKEN}
    
    Here's an example of an excellent continuation analysis for a mechanical keyboard:
    
    "From these new angles, I can now see additional important features that weren't visible before. The keyboard has a standard QWERTY layout with 104 keys including a numeric keypad on the right side. The keycaps are made of double-shot PBT plastic with a textured matte finish, and the legends are shine-through to accommodate the RGB lighting that's visible between the keys. The top plate is aluminum with a brushed finish, while the bottom case is made of textured black plastic with rubber feet at the corners. From this angle, I can see the USB-C port located at the center of the back edge, flanked by cable routing channels. There are two adjustable height feet at the rear that can be flipped up to change the typing angle."
    
    Using these new perspectives, identify features, details, or aspects of the object that weren't visible in the first two images. Build upon your previous description without repeating information you've already covered.
    """
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], pair2_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [images[2].size, images[3].size]
    
    # Generate response for second pair
    pair2_response = model.generate(
        input_ids,
        images=all_image_tensors[2:4],
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )
    pair2_output = tokenizer.batch_decode(pair2_response, skip_special_tokens=True)[0]
    descriptions.append(pair2_output)
    
    # Third pair of images with context from previous descriptions
    pair3_question = f"""Complete your analysis of the object with these final two perspectives.
    
    Your observations so far:
    First pair of angles: {pair1_output}
    
    Second pair of angles: {pair2_output}
    
    Now examining the final two angles:
    {DEFAULT_IMAGE_TOKEN} 
    {DEFAULT_IMAGE_TOKEN}
    
    Here's an example of an excellent final analysis for a decorative vase:
    
    "These final angles reveal crucial details about the vase's construction and ornamentation. The mouth of the vase has a delicate gold leaf rim that wasn't visible in previous views, and I can now confirm that the interior is glazed in a solid midnight blue that contrasts with the exterior patterns. The base has a maker's mark etched into the bottom - a small stylized flower symbol followed by what appears to be a studio signature. There's also a small unglazed ring on the bottom where the vase sits on surfaces. From these angles, I can see that the floral pattern wraps completely around the vase with no breaks in continuity, suggesting careful attention to the design. Additionally, there are three thin gold bands circling the vase: one at the rim, one at the shoulder, and one just above the base, which adds to its elegant appearance."
    
    Focus on revealing any final details that weren't visible from previous angles, and confirm or clarify any uncertain aspects of shape, color, texture, or material. Continue to ignore backgrounds and focus only on the object itself.
    """
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], pair3_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [images[4].size, images[5].size]
    
    # Generate response for third pair
    pair3_response = model.generate(
        input_ids,
        images=all_image_tensors[4:6],
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )
    pair3_output = tokenizer.batch_decode(pair3_response, skip_special_tokens=True)[0]
    descriptions.append(pair3_output)
    
    # Final comprehensive description using all previous observations AND all 6 images
    final_question = f"""Based on your complete analysis of the object from multiple angles, create a comprehensive and accurate description that captures its essential characteristics.
    
    Your detailed observations:
    First pair of angles: {pair1_output}
    Second pair of angles: {pair2_output}
    Third pair of angles: {pair3_output}

    Now you can see all six angles of the object at once:
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    
    Here's an example of an excellent final description for a vintage pocket watch:
    
    "Label: Antique Brass Pocket Watch
    Short Description: A round vintage brass pocket watch with an ornate floral engraved case, Roman numeral dial, and chain attachment.
    
    Long Description: This antique pocket watch features a circular brass case approximately 2 inches in diameter with a warm golden-brown patina. The case exterior displays intricate hand-engraved floral patterns covering the entire surface, with a small latch mechanism on the right side to open the protective cover. The watch face has a cream-colored enamel dial with black Roman numerals marking the hours and an elegant pair of blue-steel hands with ornate scrollwork. The seconds subdial is positioned at the bottom of the main dial, featuring its own miniature hand and Arabic numeral markers. The case back is hinged and opens to reveal the intricate brass mechanical movement with visible gears and ruby jewel bearings. A small winding crown sits at the 12 o'clock position, connected to the internal stem winding mechanism. The watch includes a 12-inch brass chain with alternating long and short links, ending in a decorative fob with a T-bar attachment. The brass throughout shows signs of age-appropriate wear, with darker patina in recessed areas and a subtle polish on raised surfaces, highlighting the detailed craftsmanship of this historical timepiece."
    
    Please synthesize your observations into a similar three-part description with:
    
    1. Label: A precise name for the object
    2. Short Description: A concise one-sentence description capturing the object's essence
    3. Long Description: A detailed description covering all physical aspects including:
       - Overall shape, form, and dimensions
       - Materials and surface properties
       - Colors, patterns, and visual details
       - Art style or design influences (if applicable) such as voxel/pixelized, smooth or rough, organic or geometric
       - Construction methods and component relationships
       - Functional elements or moving parts (if applicable)
       - Distinctive features that make this object unique
    
    Important guidelines:
    - Focus exclusively on the physical object itself
    - Do NOT mention the background, environment, or context
    - Do NOT refer to images, photos, angles, or views
    - Avoid phrases like "appears to be" or "seems to be" - be definitive
    - Do NOT mention that this is a 3D object or model
    - Describe the object as if it physically exists in the real world
    - Be precise about physical properties, proportions, and materials
    """
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], final_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in images]  # Get sizes for all 6 images
    
    # Generate final comprehensive response WITH all 6 images
    final_response = model.generate(
        input_ids,
        images=all_image_tensors,  # Use all images
        image_sizes=image_sizes,   # Use all image sizes
        do_sample=False,
        temperature=0,
        max_new_tokens=2048,
    )
    final_output = tokenizer.batch_decode(final_response, skip_special_tokens=True)[0]
    
    # Post-process the description to make it suitable for voxel generation
    voxel_question = f"""You are helping prepare descriptions for generating 3D voxel models. I have a detailed object description that needs to be modified to make it more suitable for voxel-based generation.

    Original description:
    {final_output}

    Please modify this description to make it better suited for generating fine-grain voxel models by:

    1. Removing all specific size information (measurements, dimensions, etc.)
    2. Removing detailed texture information (roughness, smoothness, patina, etc.)
    3. Removing material properties like weight, density, glossiness, or reflectivity
    4. Removing any references to the object's function or purpose
    5. Removing any references to the object's environment or context
    6. Removing any references to the scene's lighting or shadows
    4. Focusing on:
    - Basic geometric shape and form
    - Component relationships and structure
    - Core color information 
    - High-level visual details and patterns
    - Overall proportions (but not specific measurements)
    - Construction and connection between parts

    IMPORTANT: Format your response with exactly TWO line breaks (\\n\\n) after the Label line. This is critical for proper parsing.

    For example:
    Label: Vintage Alarm Clock

    Short Description: A classic analog alarm clock with a bell on top.

    Long Description: [rest of description...]

    Please maintain the three-part format but optimize it for voxel model generation. The result should be clean, precise, and focus on information that's useful for constructing a blocky, voxel-based representation of the object.
    """    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], voxel_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    
    # Generate voxel-optimized description (without images, just processing the text)
    voxel_response = model.generate(
        input_ids,
        do_sample=False,
        temperature=0,
        max_new_tokens=2048,
    )
    voxel_output = tokenizer.batch_decode(voxel_response, skip_special_tokens=True)[0]
    
    # Confidence evaluation step
    confidence_question = f"""You are analyzing the quality of an object description for voxel modeling. Review this voxel-optimized description along with the original set of images and rate your confidence in the description's accuracy on a scale of 1 to 10.

    Voxel-optimized description:
    {voxel_output}

    Review all angles of the object again:
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}
    {DEFAULT_IMAGE_TOKEN}

    Rate your confidence on a scale of 1 to 10, where:
    1 = The description misses major elements of the object or is fundamentally wrong
    5 = The description captures the main elements but has some inaccuracies
    10 = The description perfectly captures all important elements for voxel generation

    Consider:
    - Does the description correctly identify the object?
    - Are all key components of the object included?
    - Is the spatial relationship between components accurately described?
    - Are the colors and patterns correctly represented?
    - Would a voxel artist be able to accurately recreate this object from the description?

    Provide your confidence rating as:
    "Confidence Rating: [X/10]" 
    where X is your rating from 1 to 10.
    """
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], confidence_question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [img.size for img in images]  # Get sizes for all 6 images
    
    # Generate confidence evaluation WITH all 6 images
    confidence_response = model.generate(
        input_ids,
        images=all_image_tensors,  # Use all images again
        image_sizes=image_sizes,   # Use all image sizes
        do_sample=False,
        temperature=0,
        max_new_tokens=1024,
    )
    confidence_output = tokenizer.batch_decode(confidence_response, skip_special_tokens=True)[0]
    
    # Combine the voxel-optimized description with the confidence rating
    final_output = voxel_output + "\n\n" + confidence_output
    
    # Save output to txt file
    folder_name = folder_path.split("/")[-1]
    os.makedirs("./objaverse_descriptions", exist_ok=True)
    output_file = f"./objaverse_descriptions/{folder_name}.txt"
    
    with open(output_file, "w") as f:
        f.write(final_output)
    print(f"Output saved to {output_file}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    #before we start this process, lets remove all of the image folders that do not have 6 images in them
    # Check if the folder exists

    target_folder = "./objaverse_images"

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
            "attn_implementation": None,
        }
    overwrite_config = {}
    overwrite_config["image_aspect_ratio"] = "pad"
    llava_model_args["overwrite_config"] = overwrite_config
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, **llava_model_args)
    model.eval()
    model.to(device)


    # Use tqdm to show progress bar
    for folder in tqdm.tqdm(os.listdir(target_folder)):
        folder_path = os.path.join(target_folder, folder)
        # if the folder does not have 6 images in it, skip it
        if len(os.listdir(folder_path)) != 6:
            print(f"Skipping folder with {len(os.listdir(folder_path))} images: {folder_path}")
            continue
        image_data_augmentation(folder_path)