# coding: utf-8
import argparse
import os
import time
from dataclasses import asdict

import dgl
import numpy as np
import torch
import torch.nn.functional as F

from prism.configs.default_config import add_arguments_by_config_class, combine_args_into_config
from prism.configs.masked_mm_mgdcf_default_config import load_masked_mm_mgdcf_default_config
from prism.configs.mm_mgdcf_default_config import MMMGDCFConfig
from prism.evaluation.ranking import evaluate_mean_global_metrics
from prism.layers.mgdcf import MGDCF
from prism.layers.prism_innov import AdaptivePRISM
from prism.layers.sign import random_project
from prism.load_data import load_data
from prism.losses import compute_info_bpr_loss, compute_l2_loss
from prism.utils.data_loader_utils import create_tensors_dataloader
from prism.utils.random_utils import reset_seed


def choose_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        dgl.graph(([], []), num_nodes=1).to("cuda")
        return "cuda"
    except Exception:
        return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="baby", help="dataset name")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--method", type=str, default="prism")
    parser.add_argument("--unsmooth_coef", type=float, default=1.0)
    parser.add_argument("--gate_entropy_coef", type=float, default=1e-4)
    parser = add_arguments_by_config_class(parser, MMMGDCFConfig)
    return parser.parse_args()


def evaluate(model, g, user_embeddings, item_embeddings, v_feat, t_feat, num_users, user_items_dict, mask_user_items_dict):
    model.eval()
    with torch.no_grad():
        virtual_h = model(
            g,
            user_embeddings,
            v_feat,
            t_feat,
            item_embeddings=item_embeddings,
            return_all=False,
        )
        user_h = virtual_h[:num_users].detach().cpu().numpy()
        item_h = virtual_h[num_users:].detach().cpu().numpy()

    return evaluate_mean_global_metrics(
        user_items_dict,
        mask_user_items_dict,
        user_h,
        item_h,
        k_list=[10, 20],
        metrics=["precision", "recall", "ndcg"],
    )


def main():
    args = parse_args()
    if args.method != "prism":
        raise ValueError("Unsupported method '{}'. Use: prism.".format(args.method))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = load_masked_mm_mgdcf_default_config(args.dataset)
    config = combine_args_into_config(config, args)
    print(config)

    reset_seed(args.seed)
    device = choose_device()
    print("Using device:", device)

    (
        train_user_item_edges,
        valid_user_item_edges,
        test_user_item_edges,
        train_user_items_dict,
        train_mask_user_items_dict,
        valid_user_items_dict,
        valid_mask_user_items_dict,
        test_user_items_dict,
        test_mask_user_items_dict,
        num_users,
        num_items,
        v_feat,
        t_feat,
    ) = load_data(args.dataset)

    if config.use_rp:
        v_feat = random_project(v_feat, t_feat.size(-1))

    v_feat = v_feat.to(device)
    t_feat = t_feat.to(device)

    g = MGDCF.build_sorted_homo_graph(
        train_user_item_edges,
        num_users=num_users,
        num_items=num_items,
    ).to(device)

    embedding_size = config.embedding_size
    user_embeddings = torch.tensor(
        np.random.randn(num_users, embedding_size) / np.sqrt(embedding_size),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )
    item_embeddings = torch.tensor(
        np.random.randn(num_items, embedding_size) / np.sqrt(embedding_size),
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    model = AdaptivePRISM(
        k_e=config.k_e,
        k_t=config.k_t,
        k_v=config.k_v,
        alpha=config.alpha,
        beta=config.beta,
        input_feat_drop_rate=config.input_feat_drop_rate,
        feat_drop_rate=config.feat_drop_rate,
        user_x_drop_rate=config.user_x_drop_rate,
        item_x_drop_rate=config.item_x_drop_rate,
        edge_drop_rate=config.edge_drop_rate,
        z_drop_rate=config.z_drop_rate,
        user_in_channels=config.embedding_size,
        item_v_in_channels=v_feat.size(-1),
        item_v_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        item_t_in_channels=t_feat.size(-1),
        item_t_hidden_channels_list=[config.feat_hidden_units, embedding_size],
        bn=config.bn,
        num_clusters=config.num_clusters,
        num_samples=config.num_samples,
    ).to(device)

    optimizer = torch.optim.Adam(
        [user_embeddings, item_embeddings] + list(model.parameters()),
        lr=config.lr,
    )

    train_loader = create_tensors_dataloader(
        torch.arange(len(train_user_item_edges)),
        torch.tensor(train_user_item_edges),
        batch_size=config.batch_size,
        shuffle=True,
    )

    os.makedirs(args.result_dir, exist_ok=True)

    early_stop_metric = "recall@20"
    best_valid_score = 0.0
    best_epoch = -1
    best_valid = None
    best_test = None
    patience_count = 0

    combined_config = vars(args).copy()
    for k, v in asdict(config).items():
        combined_config[k] = v

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()
        model.train()

        for _, batch_edges in train_loader:
            with g.local_scope():
                (
                    virtual_h,
                    emb_h,
                    t_h,
                    v_h,
                    _,
                    _,
                    z_memory_h,
                    aux,
                ) = model(
                    g,
                    user_embeddings,
                    v_feat,
                    t_feat,
                    item_embeddings=item_embeddings if config.use_item_emb else None,
                    return_all=True,
                )
                user_h = virtual_h[:num_users]
                item_h = virtual_h[num_users:]

                mf_losses = compute_info_bpr_loss(
                    user_h,
                    item_h,
                    batch_edges,
                    num_negs=config.num_negs,
                    reduction="none",
                )
                l2_loss = compute_l2_loss([user_h, item_h])
                loss = mf_losses.sum() + l2_loss * config.l2_coef

                pos_user_h = user_h[batch_edges[:, 0]]
                pos_z_memory_h = z_memory_h[batch_edges[:, 1] + num_users]
                unsmooth_logits = (pos_user_h.unsqueeze(1) @ pos_z_memory_h.permute(0, 2, 1)).squeeze(1)
                unsmooth_loss = F.cross_entropy(
                    unsmooth_logits,
                    torch.zeros([batch_edges.size(0)], dtype=torch.long, device=device),
                    reduction="mean",
                )
                loss = loss + args.unsmooth_coef * unsmooth_loss

                modal_weights = aux["modal_weights"]
                gate_entropy = -(modal_weights.clamp(min=1e-9) * modal_weights.clamp(min=1e-9).log()).sum(dim=-1).mean()
                loss = loss - args.gate_entropy_coef * gate_entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for pg in optimizer.param_groups:
            new_lr = pg["lr"] * config.lr_decay
            if new_lr >= config.lr_decay_min:
                pg["lr"] = new_lr

        elapsed = time.time() - epoch_start
        print(
            "epoch={}\tloss={:.4f}\tmf={:.4f}\tl2={:.4f}\tunsmooth={:.4f}\tlr={:.6f}\ttime={:.2f}s\tpcount={}".format(
                epoch,
                float(loss.item()),
                float(mf_losses.mean().item()),
                float(l2_loss.item()),
                float(unsmooth_loss.item()),
                optimizer.param_groups[0]["lr"],
                elapsed,
                patience_count,
            )
        )

        if epoch % config.validation_freq != 0:
            continue

        print("\nEvaluation @ epoch {} ...".format(epoch))
        valid_results = evaluate(
            model,
            g,
            user_embeddings,
            item_embeddings if config.use_item_emb else None,
            v_feat,
            t_feat,
            num_users,
            valid_user_items_dict,
            valid_mask_user_items_dict,
        )
        print("valid:", valid_results)

        current = valid_results[early_stop_metric]
        if current > best_valid_score:
            test_results = evaluate(
                model,
                g,
                user_embeddings,
                item_embeddings if config.use_item_emb else None,
                v_feat,
                t_feat,
                num_users,
                test_user_items_dict,
                test_mask_user_items_dict,
            )
            best_valid_score = current
            best_epoch = epoch
            best_valid = valid_results
            best_test = test_results
            patience_count = 0
            print("test:", test_results)
        else:
            patience_count += config.validation_freq
            if patience_count >= config.patience:
                print("Early stop at epoch", epoch)
                break

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "best_epoch": best_epoch,
        "best_valid": best_valid,
        "best_test": best_test,
        "config": combined_config,
    }
    print("\nSummary:", summary)


if __name__ == "__main__":
    main()
