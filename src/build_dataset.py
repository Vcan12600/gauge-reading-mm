import json 
import random 
import os 
import copy
import torch
from torch.utils.data import Dataset
from qwen_vl_utils import process_vision_info


end_token = "<|im_end|>"
IGNORE_INDEX = -100
class InstructionDataset(Dataset):
    def __init__(self, data_path, processor, image_folder, seed):
        # Load annotation data from a JSON file, expected to contain image paths, questions, and answers
        self.ann = json.load(open(data_path))
        # Set a fixed random seed for reproducible shuffling
        random.seed(seed)
        random.shuffle(self.ann)

        # Folder containing the images
        self.image_folder = image_folder

        # Initialize text/image fields (not strictly needed here)
        self.text = ""
        self.image = ""

        # Use full dataset (could be sliced if needed)
        self.ann = self.ann[0:]

        # Processor that handles both text and image inputs (e.g., from Hugging Face)
        self.processor = processor

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        # Retrieve a single annotation
        ann = self.ann[index]
        self.image = os.path.join(self.image_folder, ann["image"])

        # Get the question text
        self.text = ann["q"]

        # Build a multimodal prompt template with both image and text
        message_template = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.image,
                    },
                    {
                        "type": "text",
                        "text": self.text
                    },
                ],
            }
        ]

        messages = [message_template]

        # Create the model input text with the answer and end token
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + ann["a"] + end_token
            for msg in messages
        ]

        # Process image and video content from the messages
        image_inputs, video_inputs = process_vision_info(messages)

        # Convert the first image to grayscale (mode "L")
        image_inputs[0] = image_inputs[0].convert("L")

        # Use the processor to tokenize text and encode image/video inputs
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt"
        )

        # Determine the length of the full input and the answer
        input_len = len(inputs["input_ids"][0])
        answer_ids = self.processor.tokenizer(ann["a"] + end_token, return_tensors="pt")["input_ids"]
        answer_len = len(answer_ids[0])

        # Construct labels: mask out everything before the answer with IGNORE_INDEX
        inputs["labels"] = copy.deepcopy(inputs["input_ids"])
        inputs["labels"][0][:input_len - answer_len] = IGNORE_INDEX

        # Convert inputs to Python lists for serialization
        inputs = {key: value.tolist() for key, value in inputs.items()}

        # Return multimodal input tensors
        return {
            "input_ids": torch.tensor(inputs["input_ids"]),
            "attention_mask": torch.tensor(inputs["attention_mask"]),
            "labels": torch.tensor(inputs["labels"]),
            "pixel_values": torch.tensor(inputs["pixel_values"]),
            "image_grid_thw": torch.tensor(inputs["image_grid_thw"])
        }
