import json
import os
from collections import Counter
from pathlib import Path


AMBER_ROOT = Path("data/AMBER")
OUT_ROOT = Path("data/amber_proto")

QUERY_DIR = AMBER_ROOT / "data" / "query"
ANN_PATH = AMBER_ROOT / "data" / "annotations.json"
IMAGE_ROOT = AMBER_ROOT / "images" / "image"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_ann_map():
    annotations = load_json(ANN_PATH)
    return {x["id"]: x for x in annotations}


def check_image(image_name):
    return str(IMAGE_ROOT / image_name)


def build_generative_only(ann_map):
    """
    Experiment 1:
    AMBER generative only.
    Concept labels:
      - safe      : truth objects
      - existence : hallucinatory target objects
    """
    queries = load_json(QUERY_DIR / "query_generative.json")
    samples = []

    for q in queries:
        ann = ann_map[q["id"]]
        image_path = check_image(q["image"])

        for obj in ann.get("truth", []):
            samples.append({
                "source": "amber_generative",
                "id": q["id"],
                "image": q["image"],
                "image_path": image_path,
                "query": q["query"],
                "concept": "safe",
                "target_word": obj,
                "annotation_type": ann["type"],
            })

        for obj in ann.get("hallu", []):
            samples.append({
                "source": "amber_generative",
                "id": q["id"],
                "image": q["image"],
                "image_path": image_path,
                "query": q["query"],
                "concept": "existence",
                "target_word": obj,
                "annotation_type": ann["type"],
            })

    return samples


def build_full_concept(ann_map):
    """
    Experiment 2:
    AMBER generative + discriminative.
    Concept labels:
      - safe
      - existence
      - attribute
      - relation
    """
    samples = []

    # generative: safe / existence
    samples.extend(build_generative_only(ann_map))

    # discriminative existence
    existence_q = load_json(QUERY_DIR / "query_discriminative-existence.json")
    for q in existence_q:
        ann = ann_map[q["id"]]
        samples.append({
            "source": "amber_discriminative",
            "id": q["id"],
            "image": q["image"],
            "image_path": check_image(q["image"]),
            "query": q["query"],
            "concept": "existence",
            "target_word": None,
            "answer": ann["truth"],
            "annotation_type": ann["type"],
        })

    # discriminative attribute: state / number / action -> attribute
    attr_q = load_json(QUERY_DIR / "query_discriminative-attribute.json")
    for q in attr_q:
        ann = ann_map[q["id"]]
        samples.append({
            "source": "amber_discriminative",
            "id": q["id"],
            "image": q["image"],
            "image_path": check_image(q["image"]),
            "query": q["query"],
            "concept": "attribute",
            "target_word": None,
            "answer": ann["truth"],
            "annotation_type": ann["type"],
        })

    # discriminative relation
    rel_q = load_json(QUERY_DIR / "query_discriminative-relation.json")
    for q in rel_q:
        ann = ann_map[q["id"]]
        samples.append({
            "source": "amber_discriminative",
            "id": q["id"],
            "image": q["image"],
            "image_path": check_image(q["image"]),
            "query": q["query"],
            "concept": "relation",
            "target_word": None,
            "answer": ann["truth"],
            "annotation_type": ann["type"],
        })

    return samples


def print_stats(name, samples):
    print(f"\n[{name}]")
    print("num samples:", len(samples))
    print("concept:", Counter(x["concept"] for x in samples))
    print("annotation_type:", Counter(x["annotation_type"] for x in samples))


def main():
    ann_map = get_ann_map()

    gen_only = build_generative_only(ann_map)
    full = build_full_concept(ann_map)

    save_json(gen_only, OUT_ROOT / "amber_proto_generative_only.json")
    save_json(full, OUT_ROOT / "amber_proto_full_concept.json")

    label_map = {
        "safe": 0,
        "existence": 1,
        "attribute": 2,
        "relation": 3,
    }
    save_json(label_map, OUT_ROOT / "label_map.json")

    print_stats("generative_only", gen_only)
    print_stats("full_concept", full)
    print(f"\nSaved to: {OUT_ROOT}")


if __name__ == "__main__":
    main()