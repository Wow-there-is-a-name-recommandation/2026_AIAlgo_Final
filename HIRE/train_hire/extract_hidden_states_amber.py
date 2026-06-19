import argparse
import gc
import json
import os
from functools import partial

import h5py
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.utils import disable_torch_init


class CustomDistributedSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas, rank, shuffle=False, seed=0, total_samples=None):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed)
        self.total_samples = total_samples if total_samples is not None else len(dataset)

    def __iter__(self):
        indices = list(super().__iter__())
        n = self.total_samples // self.num_replicas
        r = self.total_samples % self.num_replicas
        if self.rank < r:
            n += 1
        return iter(indices[:n])

    def __len__(self):
        n = self.total_samples // self.num_replicas
        r = self.total_samples % self.num_replicas
        if self.rank < r:
            n += 1
        return n


class AmberProtoDataset(Dataset):
    def __init__(self, json_path, image_root=None, start_index=0, end_index=None):
        samples = json.load(open(json_path, "r", encoding="utf-8"))

        if end_index is None:
            end_index = len(samples)

        self.start_index = start_index
        self.samples = samples[start_index:end_index]
        self.image_root = image_root

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        global_idx = self.start_index + idx

        image_path = item.get("image_path", None)
        if image_path is None:
            image_path = os.path.join(self.image_root, item["image"])

        image = Image.open(image_path).convert("RGB")

        if item.get("target_word") is not None:
            answer_text = item["target_word"]
        else:
            answer_text = "Yes" if item.get("answer", "yes") == "yes" else "No"

        return {
            "sample_idx": global_idx,
            "id": item["id"],
            "image": image,
            "image_name": item["image"],
            "query": item["query"],
            "answer_text": answer_text,
            "concept": item["concept"],
            "target_word": item.get("target_word", None),
            "source": item.get("source", ""),
            "annotation_type": item.get("annotation_type", ""),
        }


def amber_collate_fn(batch):
    return batch


extracted_buffer = {}


def extract_attn_output_hook(module, args, layer_idx):
    extracted_buffer[layer_idx] = args[0].detach().cpu()


def register_hooks(model):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(extract_attn_output_hook, layer_idx=i)
        hooks.append(target_module.register_forward_pre_hook(hook_fn))

    return hooks


def save_hidden_states(
    hs_dict,
    batch,
    input_ids,
    no_answer_lens,
    pad_token_id,
    h5_file,
    image_token_length=576,
):
    for b, item in enumerate(batch):
        sample_key = f"{item['id']}_{item['sample_idx']}"

        if sample_key in h5_file:
            continue

        no_ans_len = no_answer_lens[b]

        answer_start = image_token_length + no_ans_len - 1
        query_start = image_token_length
        query_end = answer_start

        answer_token_ids = input_ids[b, no_ans_len:]
        answer_mask = answer_token_ids != pad_token_id

        stack_answer = []
        stack_query = []

        for layer_idx in sorted(hs_dict.keys()):
            hs = hs_dict[layer_idx][b]

            ans_h = hs[answer_start:][answer_mask.cpu()]
            stack_answer.append(ans_h)

            qry_h = hs[query_start:query_end]
            if qry_h.shape[0] == 0:
                qry_h = torch.zeros(1, hs.shape[-1], dtype=hs.dtype)
            else:
                qry_h = torch.zeros(hs.shape[-1], dtype=hs.dtype)

            stack_query.append(qry_h)

        stack_answer = torch.stack(stack_answer, dim=0).to(torch.float16)
        stack_query = torch.stack(stack_query, dim=0).to(torch.float16)

        grp = h5_file.create_group(sample_key)
        grp.create_dataset("hidden_states", data=stack_answer.cpu().numpy())
        grp.create_dataset("query_hidden_states", data=stack_query.cpu().numpy())
        grp.create_dataset("token_ids", data=answer_token_ids[answer_mask].cpu().numpy())
        grp.create_dataset("query_token_ids", data=input_ids[b, :no_ans_len].cpu().numpy())

        grp.attrs["id"] = item["id"]
        grp.attrs["image"] = item["image_name"]
        grp.attrs["concept"] = item["concept"]
        grp.attrs["source"] = item["source"]
        grp.attrs["annotation_type"] = item["annotation_type"]
        grp.attrs["target_word"] = "" if item["target_word"] is None else item["target_word"]
        grp.attrs["query"] = item["query"]
        grp.attrs["answer_text"] = item["answer_text"]

    h5_file.flush()


def build_prompt(tokenizer, model, conv_mode, query, answer_text):
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_no_answer = conv.get_prompt()
    ids_no_answer = tokenizer_image_token(
        prompt_no_answer,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], answer_text)
    prompt = conv.get_prompt()
    ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )

    return ids, ids_no_answer.shape[0]


def eval_model(args):
    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
    )

    register_hooks(model)

    model = model.to("cuda")
    model.eval()

    dataset = AmberProtoDataset(
        args.data_path,
        start_index=args.start_index,
        end_index=args.end_index,
    )

    sampler = CustomDistributedSampler(
        dataset,
        num_replicas=args.num_chunks,
        rank=args.chunk_idx,
        shuffle=False,
        seed=args.seed,
        total_samples=args.data_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=amber_collate_fn,
        drop_last=False,
    )

    os.makedirs(args.hs_path, exist_ok=True)

    end_name = "end" if args.end_index is None else str(args.end_index)
    save_path = os.path.join(
        args.hs_path,
        f"hs_amber_proto_{args.split_name}_{args.start_index}_{end_name}_{args.chunk_idx}.h5",
    )

    h5_file = h5py.File(save_path, "a")

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    for batch in tqdm(dataloader):
        images = [x["image"] for x in batch]

        input_list = []
        no_answer_lens = []
        max_len = 0

        for item in batch:
            ids, no_ans_len = build_prompt(
                tokenizer=tokenizer,
                model=model,
                conv_mode=args.conv_mode,
                query=item["query"],
                answer_text=item["answer_text"],
            )
            input_list.append(ids)
            no_answer_lens.append(no_ans_len)
            max_len = max(max_len, ids.shape[0])

        padded_ids = []
        for ids in input_list:
            ids = ids.cuda()
            pad_len = max_len - ids.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len,),
                    pad_token_id,
                    dtype=ids.dtype,
                    device=ids.device,
                )
                ids = torch.cat([ids, padding], dim=0)
            padded_ids.append(ids)

        input_ids = torch.stack(padded_ids, dim=0)
        attention_mask = (input_ids != pad_token_id).long().cuda()

        image_tensor = process_images(images, image_processor, model.config)

        extracted_buffer.clear()

        with torch.no_grad():
            _ = model(
                input_ids,
                images=image_tensor.half().cuda(),
                attention_mask=attention_mask,
            )

        save_hidden_states(
            hs_dict=extracted_buffer,
            batch=batch,
            input_ids=input_ids,
            no_answer_lens=no_answer_lens,
            pad_token_id=pad_token_id,
            h5_file=h5_file,
            image_token_length=args.image_token_length,
        )

        extracted_buffer.clear()

        del input_ids
        del attention_mask
        del image_tensor
        del padded_ids
        del input_list
        del images

        torch.cuda.empty_cache()
        gc.collect()

    h5_file.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--hs-path", type=str, default="output/hidden_states_amber_proto")
    parser.add_argument("--split-name", type=str, default="full")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)

    parser.add_argument("--image-token-length", type=int, default=576)

    args = parser.parse_args()
    eval_model(args)