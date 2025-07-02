import torch
import tqdm
import torch.nn as nn
import json
import os
import random

from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from training_config import train_config
from build_dataset import InstructionDataset
from transformers import AdamW
from torch.optim.lr_scheduler import StepLR


def random_masking(x, learnable_token, mask_ratio=0.35):
    # x is assumed to have shape (L, D), where:
    #   L = sequence length
    #   D = feature dimension
    # Note: this version does NOT include a batch dimension (specific to Qwen-VL input)
    L, D = x.shape
    # Calculate the number of tokens to be masked
    
    len_chose = int(L * mask_ratio)

    # Generate random noise for each token position (used for random selection)
    noise = torch.rand(L, device=x.device)  # values in [0, 1]

    # Sort the noise to get a random permutation of indices
    ids_shuffle = torch.argsort(noise, dim=0)

    # Select the first len_chose indices as masked positions
    ids_chose = ids_shuffle[..., :len_chose]

    # Replace selected tokens with the learnable masking token
    x[ids_chose] = learnable_token

    return x

token_dim = 1280

# Create a learnable token initialized with random values
# Shape: [1, token_dim], converted to bfloat16 for memory efficiency
learnable_token = nn.Parameter(torch.randn(1, token_dim).to(torch.bfloat16))

learnable_token = learnable_token.cuda()

# Hook function to apply random masking with the learnable token
def hook_fn2(module, input, output):
    return random_masking(output, learnable_token).cuda()

# Function to calculate accuracy of meter reading predictions
# Compares the predicted value against the ground truth label
def accuracy_reading(result, label):
    err_5 = 0  
    err_10 = 0 
    result_list = result.split(" ")
    pred = result_list[0]

    if is_number(pred):
        result_f = float(pred)
        label_f = float(label["reading"])
        gauge_range = abs(float(label["max"]) - float(label["min"]))

        if abs(result_f - label_f) <= gauge_range * 0.1:
            err_10 = 1
            # Further check if it's within ±5%
            if abs(result_f - label_f) <= gauge_range * 0.05:
                err_5 = 1

    return err_5, err_10

# Utility function to check if a string can be converted to a float
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    except (TypeError, ValueError):
        pass
    return False

def eval(model, processor, label_path, image_folder, question):
    label_list = json.load(open(label_path))
    sum_err5 = 0
    sum_err10 = 0
    err_5 = 0
    err_10 = 0
    model.eval()
    for index in tqdm.tqdm(range(len(label_list)), desc='testing'):
        images = os.path.join(image_folder, label_list[index]["image"])
        message1 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image":images,
                },
                {"type": "text", "text": question},
            ],
        }
        ]   
        messages = [message1]
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        image_inputs[0] = image_inputs[0].convert("L")
        inputs = inputs.to(model.device)
        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512, temperature=0.3)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if question == "reading":
            err_5, err_10 = accuracy_reading(output_text[0], label_list[index])
            sum_err5 = sum_err5 + err_5
            sum_err10 = sum_err10 + err_10
        else:
            print("undefined qestions")
            return

    return sum_err5/len(label_list), sum_err10/len(label_list)

def train(model, train_dataloader, optimizer, processor, config):
    p = 0.35
    last_acc_zero_shot = 0
    drop_token = config.drop_token
    iseval = config.iseval
    handle = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scheduler = StepLR(optimizer, step_size=5, gamma=0.85)

    # scaler = torch.amp.GradScaler('cuda')
    for epoch in range(config.training_epoch):  # loop over the dataset multiple times
    # train
        model.train()
        train_loss = 0.0
        for index in tqdm.tqdm(range(len(train_dataloader))):
            # get the inputs
            batch = train_dataloader[index]
            for k,v in batch.items():
                batch[k] = v.to(device)


            if drop_token:
                if random.random()<p:
                    handle = model.visual.patch_embed.register_forward_hook(hook_fn2)


            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()

            if drop_token:
                if handle!= None:
                    handle.remove()
                    handle = None
        evg_loss = train_loss/len(train_dataloader)

        if iseval:
            acc_5_zero_shot, acc_10_zero_shot = eval(model, processor, config.eval_label, config.eval_image_folder)
            if acc_5_zero_shot > last_acc_zero_shot:
                model.save_pretrained(config.model_save)
                print(f"zero-shot-model saved best acc,acc:{acc_5_zero_shot}")
                last_acc_zero_shot = acc_5_zero_shot

        scheduler.step()
        print(f"this step loss:{evg_loss}, step:{epoch}")


def load_qwenvl(config):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(config.model_path, padding_side="left", min_pixels=config.min_pixels, max_pixels=config.max_pixels, do_resize=True)
    return model, processor


if __name__ == "__main__":
    model, processor = load_qwenvl(train_config)
    train_dataset = InstructionDataset(data_path=train_config.training_label, processor=processor, image_folder=train_config.training_image_folder, seed=train_config.training_seed)
    for name, param in model.named_parameters():
        if "model" in name:
            param.requires_grad = False 
    optimizer = AdamW(model.parameters(), lr=train_config.training_Lr)
    model.gradient_checkpointing_enable()
    train(model, train_dataset, optimizer, processor, train_config)