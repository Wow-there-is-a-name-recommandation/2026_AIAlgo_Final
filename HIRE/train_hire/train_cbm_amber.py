import argparse, glob, json, os, random
from collections import Counter

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


class PrototypeEncoder(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = z @ p.t()
        return sim, z, p


class ConceptBottleneck(nn.Module):
    def __init__(self, num_concepts=4, hidden_dim=64):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, proto_sim):
        concept_logits = self.bottleneck(proto_sim)
        concept_score = torch.sigmoid(concept_logits)
        return concept_logits, concept_score


class AmberH5Dataset(Dataset):
    def __init__(self, h5_dir, split_name="full", label_map=None,
                 use_query_for=("attribute", "relation"),
                 layer_mode="mean_last_k", last_k=4):
        self.paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
        if not self.paths:
            raise FileNotFoundError(f"No h5 files found in {h5_dir}")

        self.index = []
        self.use_query_for = set(use_query_for)
        self.layer_mode = layer_mode
        self.last_k = last_k

        for path in self.paths:
            with h5py.File(path, "r") as f:
                for key in f.keys():
                    concept = f[key].attrs["concept"]
                    self.index.append((path, key, concept))

        self.label_map = label_map
        if self.label_map is None:
            concepts = sorted(set(x[2] for x in self.index))
            self.label_map = {c: i for i, c in enumerate(concepts)}

        print("H5 files:", len(self.paths))
        print("Samples:", len(self.index))
        print("Concept count:", Counter(x[2] for x in self.index))
        print("Label map:", self.label_map)

    def __len__(self):
        return len(self.index)

    def pool(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x.mean(dim=1)

        if self.layer_mode == "last":
            return x[-1]
        if self.layer_mode == "mean_last_k":
            return x[-self.last_k:].mean(dim=0)
        if self.layer_mode == "mean_all":
            return x.mean(dim=0)
        raise ValueError(self.layer_mode)

    def __getitem__(self, idx):
        path, key, concept = self.index[idx]

        with h5py.File(path, "r") as f:
            g = f[key]
            if concept in self.use_query_for:
                x = g["query_hidden_states"][:]
            else:
                x = g["hidden_states"][:]

        feat = self.pool(x)
        label = self.label_map[concept]
        multi_hot = torch.zeros(len(self.label_map), dtype=torch.float32)
        multi_hot[label] = 1.0

        return feat, torch.tensor(label), multi_hot


def evaluate(proto, cbm, loader, device):
    proto.eval()
    cbm.eval()

    correct, total, loss_sum = 0, 0, 0

    with torch.no_grad():
        for x, y, multi_y in loader:
            x, y, multi_y = x.to(device), y.to(device), multi_y.to(device)

            sim, _, _ = proto(x)
            logits, score = cbm(sim)

            loss_ce = F.cross_entropy(logits, y)
            loss_bce = F.binary_cross_entropy_with_logits(logits, multi_y)
            loss = loss_ce + 0.5 * loss_bce

            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += loss.item() * y.numel()

    return loss_sum / max(total, 1), correct / max(total, 1)


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    proto_ckpt = torch.load(args.prototype_checkpoint, map_location="cpu")
    label_map = proto_ckpt["label_map"]
    proto_args = proto_ckpt.get("args", {})

    proto = PrototypeEncoder(
        input_dim=4096,
        embed_dim=proto_args.get("embed_dim", args.embed_dim),
        hidden_dim=proto_args.get("hidden_dim", args.proto_hidden_dim),
        num_classes=len(label_map),
    )
    proto.load_state_dict(proto_ckpt["model"])

    dataset = AmberH5Dataset(
        h5_dir=args.h5_dir,
        split_name=args.split_name,
        label_map=label_map,
        layer_mode=proto_args.get("layer_mode", "mean_last_k"),
        last_k=proto_args.get("last_k", 4),
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
    proto.to(device)
    cbm = ConceptBottleneck(num_concepts=len(label_map), hidden_dim=args.cbm_hidden_dim).to(device)

    if args.freeze_prototype:
        for p in proto.parameters():
            p.requires_grad = False

    params = list(cbm.parameters())
    if not args.freeze_prototype:
        params += list(proto.parameters())

    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)

    best_acc = -1

    for epoch in range(1, args.epochs + 1):
        proto.train(not args.freeze_prototype)
        cbm.train()

        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"CBM Epoch {epoch}/{args.epochs}")

        for x, y, multi_y in pbar:
            x, y, multi_y = x.to(device), y.to(device), multi_y.to(device)

            sim, _, _ = proto(x)
            logits, score = cbm(sim)

            loss_ce = F.cross_entropy(logits, y)
            loss_bce = F.binary_cross_entropy_with_logits(logits, multi_y)

            # bottleneck sparsity: 한 샘플이 한 concept만 강하게 켜지도록 유도
            loss_sparse = score.mean()

            loss = loss_ce + args.lambda_bce * loss_bce + args.lambda_sparse * loss_sparse

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

        val_loss, val_acc = evaluate(proto, cbm, val_loader, device)

        print(
            f"[{epoch:03d}] "
            f"train_loss={total_loss/max(total,1):.4f} "
            f"train_acc={correct/max(total,1):.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f}"
        )

        ckpt = {
            "prototype": proto.state_dict(),
            "cbm": cbm.state_dict(),
            "label_map": label_map,
            "args": vars(args),
            "prototype_args": proto_args,
        }

        torch.save(ckpt, os.path.join(args.out_dir, "cbm_last.pt"))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ckpt, os.path.join(args.out_dir, "cbm_best.pt"))

    with open(os.path.join(args.out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print("Best val acc:", best_acc)
    print("Saved to:", args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prototype-checkpoint", type=str,
                        default="output/checkpoint/amber_prototype_full/prototype_best.pt")
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="full")
    parser.add_argument("--out-dir", type=str, default="output/checkpoint/amber_cbm_full")

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proto-hidden-dim", type=int, default=1024)
    parser.add_argument("--cbm-hidden-dim", type=int, default=64)

    parser.add_argument("--freeze-prototype", action="store_true")

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--lambda-bce", type=float, default=0.5)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)

    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)