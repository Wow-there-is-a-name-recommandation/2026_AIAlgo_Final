import argparse
import torch
import os
import json
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.utils.data import DataLoader, DistributedSampler
from coco_data import CocoDataset, custom_collate_fn
from filelock import FileLock


def eval_model(args):
    disable_torch_init()
    device = "cuda:0"

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device, device_map=device
    )

    dataset = CocoDataset(args.image_folder, args.anno_folder, subset_size=args.data_size, seed=args.seed)
    sampler = DistributedSampler(dataset, num_replicas=args.num_chunks, rank=args.chunk_idx, seed=args.seed, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=custom_collate_fn, drop_last=False)

    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out_file:
        for image, image_names, image_path in tqdm(dataloader):
            qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
            input_ids = input_ids.repeat(len(image), 1)

            image_tensor = process_images(image, image_processor, model.config)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().to(device),
                    do_sample=False,
                    max_new_tokens=512,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            results = []
            for image_name, text in zip(image_names, outputs):
                img_id = int(image_name.split("_")[-1].split(".")[0])
                results.append(json.dumps({
                    "image_id": img_id,
                    "caption": text.split("ASSISTANT:")[-1].strip()
                }, ensure_ascii=False))

            with FileLock("baseline.lock"):
                out_file.write("\n".join(results) + "\n")
                out_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--anno-folder", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=500)
    parser.add_argument("--output-file", type=str, default="output/chair/llava_baseline.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    eval_model(args)