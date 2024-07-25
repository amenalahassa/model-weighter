import torch
import torch.nn as nn

def calculate_memory_usage(model, batch_size, input_size):
    """
    Calculate the memory usage for a given model, batch size, and input size.

    Parameters:
    - model: The neural network model (PyTorch model).
    - batch_size: The size of the batches for training/inference.
    - input_size: The size of the input tensor (e.g., (channels, height, width) for images).

    Returns:
    - ram_usage_training: Estimated RAM usage in bytes during training.
    - ram_usage_inference: Estimated RAM usage in bytes during inference.
    - gpu_usage_training: Estimated GPU memory usage in bytes during training.
    - gpu_usage_inference: Estimated GPU memory usage in bytes during inference.
    """
    # Ensure the model is on the CPU for initial RAM estimation
    model = model.cpu()

    # Create a dummy input tensor with the specified batch size and input size
    dummy_input = torch.randn(batch_size, *input_size)

    # Estimate the model size (parameters)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())

    # Estimate the forward pass activation size
    def forward_hook(module, input, output):
        activations.append(output)

    activations = []
    hooks = []
    for layer in model.modules():
        if isinstance(layer, nn.Module):
            hooks.append(layer.register_forward_hook(forward_hook))

    # Perform a forward pass to populate the activations list
    model(dummy_input)

    # Remove the hooks
    for hook in hooks:
        hook.remove()

    activation_size = sum(act.numel() * act.element_size() for act in activations)

    # Estimate the backward pass activation size (same as forward pass)
    backward_activation_size = activation_size

    # Total RAM usage during training (model parameters + activations for forward and backward pass)
    ram_usage_training = param_size + (activation_size + backward_activation_size)

    # Total RAM usage during inference (model parameters + activations for forward pass only)
    ram_usage_inference = param_size + activation_size

    # GPU usage estimation (if using a GPU)
    if torch.cuda.is_available():
        model = model.cuda()
        dummy_input = dummy_input.cuda()

        # Clear any existing memory
        torch.cuda.empty_cache()

        # Perform a forward pass to estimate GPU memory usage during inference
        gpu_memory_before = torch.cuda.memory_allocated()
        output = model(dummy_input)
        gpu_memory_after = torch.cuda.memory_allocated()

        gpu_usage_inference = gpu_memory_after - gpu_memory_before

        # Perform a backward pass to estimate GPU memory usage during training
        loss = output.sum()
        gpu_memory_before = torch.cuda.memory_allocated()
        loss.backward()
        gpu_memory_after = torch.cuda.memory_allocated()

        gpu_usage_training = gpu_memory_after - gpu_memory_before

        # Free GPU resources
        del dummy_input
        del output
        del loss
        model = model.cpu()
        torch.cuda.empty_cache()
    else:
        gpu_usage_training = None
        gpu_usage_inference = None

    return ram_usage_training, ram_usage_inference, gpu_usage_training, gpu_usage_inference
