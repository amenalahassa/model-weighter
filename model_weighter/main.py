import torch
import torch.nn as nn
from helpers import calculate_memory_usage

if __name__ == "__main__":
    # Define a simple model for demonstration
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc1 = nn.Linear(784, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, 10)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    model = SimpleModel()
    batch_size = 32
    input_size = (1, 28, 28)  # Example for MNIST dataset

    ram_usage_training, ram_usage_inference, gpu_usage_training, gpu_usage_inference = calculate_memory_usage(model, batch_size, input_size)

    print(f"Estimated RAM usage during training: {ram_usage_training / (1024 ** 2):.2f} MB")
    print(f"Estimated RAM usage during inference: {ram_usage_inference / (1024 ** 2):.2f} MB")
    print(f"Estimated GPU usage during training: {gpu_usage_training / (1024 ** 2):.2f} MB" if gpu_usage_training else "GPU not available")
    print(f"Estimated GPU usage during inference: {gpu_usage_inference / (1024 ** 2):.2f} MB" if gpu_usage_inference else "GPU not available")