from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from PIL import Image
import requests
import copy
import torch
import sys
import warnings
warnings.filterwarnings("ignore")

class MultiImageLLaVA:
    def __init__(self, pretrained="lmms-lab/llava-onevision-qwen2-7b-ov", model_name="llava_qwen", 
                 device="cuda", device_map="auto"):
        self.device = device
        llava_model_args = {
            "multimodal": True,
            "attn_implementation": None,
        }
        
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            pretrained, None, model_name, device_map=device_map, **llava_model_args
        )
        self.model.eval()
        
    def load_images_from_urls(self, urls):
        """Load multiple images from URLs."""
        images = []
        for url in urls:
            try:
                img = Image.open(requests.get(url, stream=True).raw)
                images.append(img)
            except Exception as e:
                print(f"Error loading image from {url}: {e}")
        return images
    
    def load_images_from_paths(self, paths):
        """Load multiple images from local file paths."""
        images = []
        for path in paths:
            try:
                img = Image.open(path)
                images.append(img)
            except Exception as e:
                print(f"Error loading image from {path}: {e}")
        return images
    
    def process_images_batch(self, images, question, conv_template="qwen_1_5"):
        """Process multiple images and generate responses."""
        # Process all images
        image_tensor = process_images(images, self.image_processor, self.model.config)
        image_tensor = [img.to(dtype=torch.float16, device=self.device) for img in image_tensor]
        
        # Prepare question for each image
        responses = []
        for i in range(len(images)):
            # Create conversation template for each image
            conv = copy.deepcopy(conv_templates[conv_template])
            current_question = DEFAULT_IMAGE_TOKEN + f"\n{question}"
            conv.append_message(conv.roles[0], current_question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize input
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                IMAGE_TOKEN_INDEX, 
                return_tensors="pt"
            ).unsqueeze(0).to(self.device)
            
            # Generate response
            image_sizes = [images[i].size]
            output = self.model.generate(
                input_ids,
                images=[image_tensor[i]],  # Pass single image tensor
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=4096,
            )
            
            # Decode response
            response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            responses.append(response)
            
        return responses

# Example usage
def main():
    # Initialize the model
    llava = MultiImageLLaVA()
    
    # Example with URLs
    urls = [
        "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true",
        "https://raw.githubusercontent.com/haotian-liu/LLaVA/main/images/llava_logo.png"
    ]
    images = llava.load_images_from_urls(urls)
    
    # Or with local paths
    # paths = ["path/to/image1.jpg", "path/to/image2.jpg"]
    # images = llava.load_images_from_paths(paths)
    
    # Process all images with a question
    question = "What is shown in this image?"
    responses = llava.process_images_batch(images, question)
    
    # Print responses
    for i, response in enumerate(responses):
        print(f"\nImage {i+1} Response:")
        print(response)

if __name__ == "__main__":
    main()