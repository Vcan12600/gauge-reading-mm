from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import argparse
# default: Load the model on the available device(s)
def load_qwenvl(model_path):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, padding_side="left", min_pixels=config.min_pixels, max_pixels=config.max_pixels, do_resize=True)
    return model, processor

def inference(text, image, processor, model):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {"type": "text", "text": text},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


def main():
    parser = argparse.ArgumentParser(description="Gauge inference utility")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model weights",
        default=""
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the input image",
        default=""
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Type of inference: 'reading' for direct value reading, 'norm' for normalized value",
        default="reading"
    )
    
    args = parser.parse_args()

    if args.model_path == "":
        print("Error: Please provide the path to the model weights using --model_path.")
        return

    if args.image == "":
        print("Error: Please provide the path to the input image using --image.")
        return

    if args.prompt not in ["reading", "norm"]:
        print("Error: Invalid prompt. Please specify either 'reading' or 'norm' using --prompt.")
        return
    model, processor = load_qwenvl(args.model_path)
    inference(args.prompt, args.image, processor, model)

if __name__ == "__main__":
    main()

