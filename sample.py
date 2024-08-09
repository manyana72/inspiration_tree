import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline

# Define paths and variables
output_path = '/scratch/m_goyal1.iitr/inspiration_tree/input_concepts/buddha/v1'
model_id = 'runwayml/stable-diffusion-v1-5'
path_to_embed = '/scratch/m_goyal1.iitr/inspiration_tree/outputs/buddha_v*_0.5/v0/v0_seed0/learned_embeds-steps-200.bin'
device = torch.device("cuda")

# Ensure output directory exists
final_samples_path = f"{output_path}/final_samples"
if not os.path.exists(final_samples_path):
    os.mkdir(final_samples_path)


# Define prompts and load embeddings
prompts_template = [
    "An image of a {}",
    "A photo of a {}",
    "A close-up photo of a {}",
    "A good photo of a {}",
    "A clear image of a {}",
    "The image of a {}"
]
prompt_to_vec = {}
assert os.path.exists(path_to_embed)
data = torch.load(path_to_embed)
combined = []
prompts = []
for w_ in data.keys():
    prompt_to_vec[w_] = data[w_]
    combined.append(w_)
    prompts.append(w_)
prompts.append(" ".join(combined))

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False).to(device)


def load_tokens(pipe, data, device):
    """
    Adds the new learned tokens into the predefined dictionary of pipe.
    """
    added_tokens = []
    for t_ in data.keys():
        added_tokens.append(t_)
    num_added_tokens = pipe.tokenizer.add_tokens(added_tokens)
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    for token_ in data.keys():
        ref_token = pipe.tokenizer.tokenize(token_)
        ref_indx = pipe.tokenizer.convert_tokens_to_ids(ref_token)[0]
        embd_cur = data[token_].to(device).to(dtype=torch.float16)
        pipe.text_encoder.text_model.embeddings.token_embedding.weight[ref_indx] = embd_cur



load_tokens(pipe, prompt_to_vec, device)

print("Prompts loaded to pipe ...")
print(prompt_to_vec.keys())

# Set seeds and number of images
gen_seeds = [4321, 95, 11, 87654, 5678, 1234]  # 6 unique seeds

# Generate and save images
for i, (base_prompt, gen_seed) in enumerate(zip(prompts_template, gen_seeds)):
    with torch.no_grad():
        torch.manual_seed(gen_seed)
        for prompt in prompts:
            formatted_prompt = base_prompt.format(prompt)
            images = pipe(prompt=[formatted_prompt], num_inference_steps=25, guidance_scale=7.5).images

            # Save each image separately
            for j, image in enumerate(images):
                image_path = os.path.join(final_samples_path, f"sample_{i + 1}_{j + 1}.jpg")
                image.save(image_path)
                print(f"Saved image {image_path}")
