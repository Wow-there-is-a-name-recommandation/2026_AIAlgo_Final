import argparse
import glob
import json
import os
import random
from collections import Counter

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class AmberGenerativeExistenceDataset(Dataset):
    """
    AMBER generative 샘플 중 safe / existence만 사용.
    hidden_states는 answer_text == target_word인 target token 표현으로 가정.

    사용 샘플:
      annotation_type == "generative"
      concept in {"safe", "existence"}
      target_word != ""
      answer_text.lower() == target_word.lower()
    """

    def __init__(
        self,
        h5_dir,
        split_name="full",
        layer_mode="mean_last_k",
        last_k=4,
        allowed_concepts=("safe", "existence"),
    ):
        self.paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
        if not self.paths:
            raise FileNotFoundError(f"No h5 files found: {h5_dir}/hs_amber_proto_{split_name}_*_0.h5")

        self.index = []
        self.layer_mode = layer_mode
        self.last_k = last_k
        self.allowed_concepts = set(allowed_concepts)

        raw_counter = Counter()
        kept_counter = Counter()
        skip_counter = Counter()

        for path in self.paths:
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    g = f[key]

                    concept = str(g.attrs.get("concept", ""))
                    ann = str(g.attrs.get("annotation_type", ""))
                    answer_text = str(g.attrs.get("answer_text", "")).strip().lower()
                    target_word = str(g.attrs.get("target_word", "")).strip().lower()

                    raw_counter[(concept, ann)] += 1

                    if ann != "generative":
                        skip_counter["not_generative"] += 1
                        continue

                    if concept not in self.allowed_concepts:
                        skip_counter["concept_not_allowed"] += 1
                        continue

                    if target_word == "":
                        skip_counter["empty_target_word"] += 1
                        continue

                    if answer_text != target_word:
                        skip_counter["answer_target_mismatch"] += 1
                        continue

                    if "hidden_states" not in g:
                        skip_counter["missing_hidden_states"] += 1
                        continue

                    self.index.append((path, key, concept))
                    kept_counter[concept] += 1

        self.concepts = sorted(list(self.allowed_concepts))
        self.label_map = {c: i for i, c in enumerate(self.concepts)}

        print("H5 files:", len(self.paths))
        print("Raw count:", raw_counter)
        print("Kept samples:", len(self.index))
        print("Kept concept count:", kept_counter)
        print("Skipped:", skip_counter)
        print("Label map:", self.label_map)

        if len(self.index) == 0:
            raise RuntimeError("No usable generative safe/existence samples found.")

    def __len__(self):
        return len(self.index)

    def _pool_layer(self, x):
        """
        x: [L, T, H]
        T는 target_word의 subword token 길이.
        return: [H]
        """
        x = torch.tensor(x, dtype=torch.float32)

        if x.ndim != 3:
            raise ValueError(f"Expected hidden_states [L,T,H], got {tuple(x.shape)}")

        # target_word가 여러 subword로 나뉜 경우 평균
        x = x.mean(dim=1)  # [L, H]

        if self.layer_mode == "last":
            return x[-1]

        if self.layer_mode == "mean_last_k":
            return x[-self.last_k:].mean(dim=0)

        if self.layer_mode == "mean_all":
            return x.mean(dim=0)

        raise ValueError(f"Unknown layer_mode: {self.layer_mode}")

    def __getitem__(self, idx):
        path, key, concept = self.index[idx]

        with h5py.File(path, "r") as f:
            g = f[key]
            x = g["hidden_states"][:]
            answer_text = str(g.attrs.get("answer_text", ""))
            target_word = str(g.attrs.get("target_word", ""))

        feat = self._pool_layer(x)
        label = self.label_map[concept]

        meta = {
            "concept": concept,
            "answer_text": answer_text,
            "target_word": target_word,
            "path": path,
            "key": key,
        }

        return feat, torch.tensor(label, dtype=torch.long), meta


class ProtoNet(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = z @ p.t()
        return logits, z, p


def prototype_losses(logits, z, p, y, margin=0.2):
    loss_cls = F.cross_entropy(logits, y)

    target_p = p[y]
    loss_cluster = (1.0 - (z * target_p).sum(dim=-1)).mean()

    sim = z @ p.t()
    pos_sim = sim.gather(1, y[:, None])

    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.scatter_(1, y[:, None], False)
    neg_sim = sim.masked_select(mask).view(sim.size(0), -1)

    loss_sep = F.relu(margin + neg_sim - pos_sim).mean()

    return loss_cls, loss_cluster, loss_sep


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    pred_counter = Counter()
    label_counter = Counter()

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits, z, p = model(x)
            loss_cls, loss_cluster, loss_sep = prototype_losses(logits, z, p, y)
            loss = loss_cls + 0.1 * loss_cluster + 0.1 * loss_sep

            pred = logits.argmax(dim=-1)

            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()

            pred_counter.update(pred.cpu().tolist())
            label_counter.update(y.cpu().tolist())

    return loss_sum / max(total, 1), correct / max(total, 1), pred_counter, label_counter


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = AmberGenerativeExistenceDataset(
        h5_dir=args.h5_dir,
        split_name=args.split_name,
        layer_mode=args.layer_mode,
        last_k=args.last_k,
    )

    val_size = int(len(dataset) * args.val_ratio)
    train_size = len(dataset) - val_size

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProtoNet(
        input_dim=4096,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(dataset.label_map),
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            logits, z, p = model(x)

            loss_cls, loss_cluster, loss_sep = prototype_losses(logits, z, p, y)
            loss = loss_cls + args.lambda_cluster * loss_cluster + args.lambda_sep * loss_sep

            optim.zero_grad()
            loss.backward()
            optim.step()

            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            total_loss += loss.item() * y.numel()

            pbar.set_postfix({
                "loss": total_loss / max(total, 1),
                "acc": correct / max(total, 1),
            })

        val_loss, val_acc, pred_counter, label_counter = evaluate(model, val_loader, device)

        print(
            f"[{epoch:03d}] "
            f"train_loss={total_loss / max(total, 1):.4f} "
            f"train_acc={correct / max(total, 1):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )
        print("val label counter:", label_counter)
        print("val pred counter :", pred_counter)

        ckpt = {
            "model": model.state_dict(),
            "label_map": dataset.label_map,
            "concepts": dataset.concepts,
            "args": vars(args),
        }

        torch.save(ckpt, os.path.join(args.out_dir, "prototype_last.pt"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "prototype_best.pt"))

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.label_map, f, indent=2, ensure_ascii=False)

    print("Best val acc:", best_acc)
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="full")
    parser.add_argument("--out-dir", type=str, default="output/checkpoint/amber_prototype_existence_generative")

    parser.add_argument("--layer-mode", type=str, default="mean_last_k", choices=["last", "mean_last_k", "mean_all"])
    parser.add_argument("--last-k", type=int, default=4)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--lambda-cluster", type=float, default=0.1)
    parser.add_argument("--lambda-sep", type=float, default=0.1)

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)