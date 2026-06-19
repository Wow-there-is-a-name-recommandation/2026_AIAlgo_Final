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


class AmberGenerativeExistenceLayerDataset(Dataset):
    def __init__(
        self,
        h5_dir,
        split_name="generative_only",
        allowed_concepts=("safe", "existence"),
    ):
        self.paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
        if not self.paths:
            raise FileNotFoundError(f"No h5 files found: {h5_dir}/hs_amber_proto_{split_name}_*_0.h5")

        self.index = []
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
            raise RuntimeError("No usable samples found.")

    def __len__(self):
        return len(self.index)

    def _to_layerwise_feature(self, x):
        """
        x: [L, T, H]
        return: [L, H]

        T는 target_word가 tokenizer에서 나뉜 subword 수.
        layer 평균은 하지 않고, subword 평균만 수행한다.
        """
        x = torch.tensor(x, dtype=torch.float32)

        if x.ndim != 3:
            raise ValueError(f"Expected hidden_states [L,T,H], got {tuple(x.shape)}")

        x = x.mean(dim=1)  # [L,H]
        return x

    def __getitem__(self, idx):
        path, key, concept = self.index[idx]

        with h5py.File(path, "r") as f:
            g = f[key]
            x = g["hidden_states"][:]
            answer_text = str(g.attrs.get("answer_text", ""))
            target_word = str(g.attrs.get("target_word", ""))

        feat = self._to_layerwise_feature(x)
        label = self.label_map[concept]

        meta = {
            "concept": concept,
            "answer_text": answer_text,
            "target_word": target_word,
            "path": path,
            "key": key,
        }

        return feat, torch.tensor(label, dtype=torch.long), meta


class LayerWiseProtoNet(nn.Module):
    """
    입력:
      x: [B, L, 4096]

    출력:
      logits: [B, C]
        layer별 prototype similarity를 평균한 최종 분류 logit

      sim: [B, L, C]
        각 layer에서 각 prototype class와의 similarity

      z: [B, L, D]
        각 layer의 embedding

      p: [L, C, D]
        layer-wise prototype bank
    """
    def __init__(
        self,
        input_dim=4096,
        embed_dim=256,
        hidden_dim=1024,
        num_classes=2,
        num_layers=32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.prototypes = nn.Parameter(
            torch.randn(num_layers, num_classes, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: [B,L,H]
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,L,H], got {tuple(x.shape)}")

        if x.size(1) != self.num_layers:
            raise ValueError(f"Expected num_layers={self.num_layers}, got {x.size(1)}")

        z = self.encoder(x)                  # [B,L,D]
        z = F.normalize(z, dim=-1)

        p = F.normalize(self.prototypes, dim=-1)  # [L,C,D]

        sim = torch.einsum("bld,lcd->blc", z, p)  # [B,L,C]

        # 최종 class 판단은 layer별 evidence 평균
        logits = sim.mean(dim=1)  # [B,C]

        return logits, sim, z, p


def layerwise_prototype_losses(logits, sim, z, p, y, margin=0.2):
    """
    logits: [B,C]
    sim   : [B,L,C]
    z     : [B,L,D]
    p     : [L,C,D]
    y     : [B]
    """
    loss_cls = F.cross_entropy(logits, y)

    B, L, D = z.shape

    # 각 샘플의 정답 class prototype을 모든 layer에서 선택
    # p[:, y, :] -> [L,B,D], permute -> [B,L,D]
    target_p = p[:, y, :].permute(1, 0, 2)

    loss_cluster = (1.0 - (z * target_p).sum(dim=-1)).mean()

    # positive similarity: [B,L,1]
    pos_sim = sim.gather(
        dim=2,
        index=y[:, None, None].expand(-1, L, 1)
    )

    # negative similarity: [B,L,C-1]
    mask = torch.ones_like(sim, dtype=torch.bool)
    mask.scatter_(2, y[:, None, None].expand(-1, L, 1), False)
    neg_sim = sim.masked_select(mask).view(B, L, -1)

    loss_sep = F.relu(margin + neg_sim - pos_sim).mean()

    return loss_cls, loss_cluster, loss_sep


def evaluate(model, loader, device, lambda_cluster=0.1, lambda_sep=0.1):
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

            logits, sim, z, p = model(x)
            loss_cls, loss_cluster, loss_sep = layerwise_prototype_losses(logits, sim, z, p, y)

            loss = loss_cls + lambda_cluster * loss_cluster + lambda_sep * loss_sep

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

    dataset = AmberGenerativeExistenceLayerDataset(
        h5_dir=args.h5_dir,
        split_name=args.split_name,
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

    model = LayerWiseProtoNet(
        input_dim=4096,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_classes=len(dataset.label_map),
        num_layers=args.num_layers,
    ).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"LayerProto Epoch {epoch}/{args.epochs}")

        for x, y, _ in pbar:
            x = x.to(device)  # [B,L,H]
            y = y.to(device)

            logits, sim, z, p = model(x)

            loss_cls, loss_cluster, loss_sep = layerwise_prototype_losses(
                logits, sim, z, p, y, margin=args.margin
            )

            loss = (
                loss_cls
                + args.lambda_cluster * loss_cluster
                + args.lambda_sep * loss_sep
            )

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
                "cls": loss_cls.item(),
                "clu": loss_cluster.item(),
                "sep": loss_sep.item(),
            })

        val_loss, val_acc, pred_counter, label_counter = evaluate(
            model,
            val_loader,
            device,
            lambda_cluster=args.lambda_cluster,
            lambda_sep=args.lambda_sep,
        )

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
            "model_type": "layerwise_prototype",
            "num_layers": args.num_layers,
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
    parser.add_argument("--split-name", type=str, default="generative_only")
    parser.add_argument("--out-dir", type=str, default="output/checkpoint/amber_layerwise_prototype_existence_generative")

    parser.add_argument("--num-layers", type=int, default=32)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--lambda-cluster", type=float, default=0.1)
    parser.add_argument("--lambda-sep", type=float, default=0.1)

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)