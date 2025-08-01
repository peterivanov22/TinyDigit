import torch
import random 
from tqdm.auto import tqdm as auto_tqdm 

#----- method from tuna

def create_training_data_from_dataset(quantized_dataset, context_length, start_token_int=256, max_samples=100000, max_images_to_process=None, random_drop=0.8):
    """
    Create training data (context tokens, target token) for Model V1 directly from a quantized Dataset.
    
    Args:
        quantized_dataset: PyTorch Dataset object yielding quantized image tensors.
        context_length (int): Size of context window.
        start_token_int (int): Integer value for the start/padding token.
        num_pixel_values (int): The number of actual pixel values (K6).
        max_samples (int, optional): Maximum number of (context,target) training pairs to generate.
        max_images_to_process (int, optional): Limit the number of images from the dataset to process. Defaults to all.
    
    Returns:
        contexts (Tensor): [N_SAMPLES, context_length] of integer tokens.
        targets (Tensor): [N_SAMPLES] of integer target tokens (0 to K-1).
    """
    all_contexts = []
    all_targets = []
    samples_collected = 0
    
    num_images_to_process = len(quantized_dataset)
    if max_images_to_process is not None:
        num_images_to_process = min(num_images_to_process, max_images_to_process)

    print(f"Generating V1 training data from {num_images_to_process} images (max {max_samples:,} samples)...")

    # Iterate directly over the Dataset object
    pbar_images = auto_tqdm(range(num_images_to_process), desc="Processing Images for V1 Data")
    for i in pbar_images:
        if samples_collected >= max_samples:
            pbar_images.set_description(f"Max samples ({max_samples}) reached. Stopping image processing.")
            break
        
        image_tensor, _ = quantized_dataset[i] # Get i-th image (already quantized)
        # quantized_image_tensor shape is [C, H, W], e.g., [3, 224, 224]
        
        flat_token_image = image_tensor.view(3, -1) # Flatten to [3, N_PIXELS]
        n_pixels = flat_token_image.shape[1]
            
        # Padded sequence for context building
        pad = torch.full((3, context_length), start_token_int, dtype = torch.long)

        #(pad.shape)
        #print(flat_token_image.shape)

        padded_token_sequence = torch.cat(
            (pad, flat_token_image), # Should already be .long from quantization
            dim=1
        )
            
        for pixel_idx in range(n_pixels-context_length):
            if samples_collected >= max_samples:
                break # Break inner loop
            
            if random.random() > random_drop:
                context = flat_token_image[:,pixel_idx : pixel_idx + context_length] #(3, context_length)
                target = flat_token_image[:, pixel_idx+1 : pixel_idx+context_length+1] #(3,context_length)?
                
                    
                all_contexts.append(context)    
                all_targets.append(target) #(1,3,1)? do we setill need this?
                samples_collected += 1
    
    pbar_images.close() # Close the progress bar for images

    if not all_contexts:
        print("Warning: No training samples collected. Check max_samples or dataset processing.")
        # Return empty tensors with correct number of dimensions to avoid errors later
        return torch.empty((0, context_length), dtype=torch.long), torch.empty((0), dtype=torch.long)

    contexts_tensor = torch.stack(all_contexts).long()
    targets_tensor = torch.stack(all_targets).long()
    
    indices = torch.randperm(len(contexts_tensor))
    contexts_tensor = contexts_tensor[indices]
    contexts_tensor = contexts_tensor.permute(0,2,1)
    targets_tensor = targets_tensor[indices]
    targets_tensor = targets_tensor.permute(0,2,1)
    
    print(f"Generated {len(contexts_tensor):,} V1 training pairs.")
    return contexts_tensor, targets_tensor