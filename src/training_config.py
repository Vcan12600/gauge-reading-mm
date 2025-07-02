from dataclasses import dataclass, field
from typing import List

@dataclass
class train_config:
    # Path to the model architecture or weights
    model_path: str = "../model"
    
    # Directory where the trained model will be saved
    model_save: str = "../save"
    
    # Minimum number of pixels allowed for input images
    min_pixels: int = 256 * 28 * 28
    
    # Maximum number of pixels allowed for input images
    max_pixels: int = 1280 * 28 * 28
    
    # Number of training epochs
    training_epoch: int = 20
    
    # Initial learning rate for training
    training_Lr: float = 1e-5
    
    # Path to the training label file (in JSON format)
    training_label: str = "../dataset/train/label.json"
    
    # Directory containing training images
    training_image_folder: str = "../dataset/train/images"
    
    # Random seed for reproducibility
    training_seed: int = 22
    
    # Step size for the learning rate scheduler (i.e., how often to decay)
    scheduler_step_size: int = 5
    
    # Gamma value for learning rate decay (multiplier)
    scheduler_gamma: float = 0.85
    
    # Whether to enable token dropping strategy (for visual-language fusion)
    drop_token: bool = True
    
    # Whether to run evaluation after training
    iseval: bool = True
    
    # Path to the evaluation label file (in JSON format)
    eval_label: str = "../dataset/eval/label.json"
    
    # Directory containing evaluation images
    eval_image_folder: str = "../dataset/eval/images"
    
    # Prompt keyword used to instruct the visual-language model
    # to output the actual (real) meter reading instead of a normalized value
    q1: str = "reading"
