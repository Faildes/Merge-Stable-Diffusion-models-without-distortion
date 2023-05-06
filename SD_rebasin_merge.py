import argparse
import torch
import os
import safetensors.torch
import tqdm

from weight_matching import sdunet_permutation_spec, weight_matching, apply_permutation


parser = argparse.ArgumentParser(description= "Merge two stable diffusion models with git re-basin")
parser.add_argument("model_a", type=str, help="Path to model a")
parser.add_argument("model_b", type=str, help="Path to model b")
parser.add_argument("--vae", type=str, help="Path to vae", default=None, required=False)
parser.add_argument("--device", type=str, help="Device to use, defaults to cpu", default="cpu", required=False)
parser.add_argument("--output", type=str, help="Output file name, without extension", default="merged", required=False)
parser.add_argument("--usefp16", type=bool, help="Whether to use half precision", default=True, required=False)
parser.add_argument("--save_safetensors", type=bool, help="Whether to save as .safetensors", default=False, required=False)
parser.add_argument("--alpha", type=str, help="Ratio of model A to B", default="0.5", required=False)
parser.add_argument("--iterations", type=str, help="Number of steps to take before reaching alpha", default="10", required=False)
args = parser.parse_args()   
device = args.device

def flatten_params(model):
  global device
  k = read_state_dict(model, map_location=device)
  return k

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

checkpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

def transform_checkpoint_dict_key(k):
  for text, replacement in checkpoint_dict_replacements.items():
      if k.startswith(text):
          k = replacement + k[len(text):]

  return k

def get_state_dict_from_checkpoint(pl_sd):
  pl_sd = pl_sd.pop("state_dict", pl_sd)
  pl_sd.pop("state_dict", None)

  sd = {}
  for k, v in pl_sd.items():
      new_key = transform_checkpoint_dict_key(k)

      if new_key is not None:
          sd[new_key] = v

  pl_sd.clear()
  pl_sd.update(sd)

  return pl_sd

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
  _, extension = os.path.splitext(checkpoint_file)
  if extension.lower() == ".safetensors":
      device = map_location
      pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
  else:
      pl_sd = torch.load(checkpoint_file, map_location=map_location)

  if print_global_state and "global_step" in pl_sd:
      print(f"Global Step: {pl_sd['global_step']}")

  sd = get_state_dict_from_checkpoint(pl_sd)
  return sd

_, extension_a = os.path.splitext(args.model_a)
if extension_a.lower() == ".safetensors":
    model_a = safetensors.torch.load_file(args.model_a, device=device)
else:
    model_a = torch.load(args.model_a, map_location=device)

_, extension_b = os.path.splitext(args.model_b)
if extension_b.lower() == ".safetensors":
    model_b = safetensors.torch.load_file(args.model_b, device=device)
else:
    model_b = torch.load(args.model_b, map_location=device)

if args.vae is not None:
  _, extension_vae = os.path.splitext(args.vae)
  if extension_vae.lower() == ".safetensors":
      vae = safetensors.torch.load_file(args.vae, device=device)
  else:
      vae = torch.load(args.vae, map_location=device)

theta_0 = read_state_dict(args.model_a, map_location=device)
theta_0_a = theta_0
theta_1 = read_state_dict(args.model_b, map_location=device)
theta_1_a = theta_1

alpha = float(args.alpha)
iterations = int(args.iterations)
step = alpha/iterations
permutation_spec = sdunet_permutation_spec()
special_keys = ["first_stage_model.decoder.norm_out.weight", "first_stage_model.decoder.norm_out.bias", "first_stage_model.encoder.norm_out.weight", 
"first_stage_model.encoder.norm_out.bias", "model.diffusion_model.out.0.weight", "model.diffusion_model.out.0.bias"]

if args.usefp16:
    print("Using half precision")
else:
    print("Using full precision")

for x in range(iterations):
    print(f"""
    ---------------------
         ITERATION {x+1}
    ---------------------
    """)

    # In order to reach a certain alpha value with a given number of steps,
    # You have to calculate an alpha for each individual iteration
    if x > 0:
        new_alpha = 1 - (1 - step*(1+x)) / (1 - step*(x))
    else:
        new_alpha = step
    print(f"new alpha = {new_alpha}\n")

    for key in tqdm.tqdm(theta_1.keys(),desc="Merging:"):
      if "model" in key and key in theta_1:
        theta_0[key] = (1 - (new_alpha)) * theta_0[key] + (new_alpha) * theta_1[key]

    if x == 0:
        for key in theta_1.keys():
            if "model" in key and key not in theta_0:
                theta_0[key] = theta_1[key]

    print("FINDING PERMUTATIONS")

    # Replace theta_0 with a permutated version using model A and B    
    first_permutation, y = weight_matching(permutation_spec, theta_0_a, theta_0, usefp16=args.usefp16)
    theta_0 = apply_permutation(permutation_spec, first_permutation, theta_0)
    second_permutation, z = weight_matching(permutation_spec, theta_1_a, theta_0, usefp16=args.usefp16)
    theta_3= apply_permutation(permutation_spec, second_permutation, theta_0)

    new_alpha = torch.nn.functional.normalize(torch.sigmoid(torch.Tensor([y, z])), p=1, dim=0).tolist()[0]

    # Weighted sum of the permutations
    
    for key in special_keys:
        theta_0[key] = (1 - new_alpha) * (theta_0[key]) + (new_alpha) * (theta_3[key])
if args.save_safetensors:
  output_file = f'{args.output}.safetensors'
else:
  output_file = f'{args.output}.ckpt'

# check if output file already exists, ask to overwrite
if os.path.isfile(output_file):
    print("Output file already exists. Overwrite? (y/n)")
    while True:
        overwrite = input()
        if overwrite == "y":
            break
        elif overwrite == "n":
            print("Exiting...")
            exit()
        else:
            print("Please enter y or n")

print("\nSaving...")
if args.save_safetensors:
  safetensors.torch.save_file(theta_0, output_file, metadata={"format": "pt"})
else:
  torch.save({"state_dict": theta_0}, output_file)

print("Done!")
