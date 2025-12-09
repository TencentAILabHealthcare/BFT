## Balanced Fine-Tuning (BFT)

This repository extends LLaMA-Factory by adding Balanced Fine-Tuning (BFT), a simple and effective improvement over Supervised Fine-Tuning (SFT) and Dynamic Fine-Tuning (DFT). BFT preserves DFT’s stability while shifting focus toward underconfident samples for better robustness and generalization.

---

## Environment Setup

Use the following script to install the CLI and required dependencies (uv, DeepSpeed, bitsandbytes, TensorBoard, Transformers pin, PATH setup). Please replace `/xxx/LLaMA-Factory` with your actual path.

```bash
# File: env_llama_factory.sh

cd LLaMA-Factory

curl -LsSf https://astral.sh/uv/install.sh | sh

source $HOME/.local/bin/env

cd /xxx/LLaMA-Factory

uv tool install -e .

export TOOLBIN="$(dirname "$(readlink -f "$(command -v llamafactory-cli)")")"

export PATH="$TOOLBIN:$PATH"

"$TOOLBIN/python" -m ensurepip --upgrade

"$TOOLBIN/python" -m pip --version

"$TOOLBIN/python" -m pip install "deepspeed>=0.10.0,<0.17" bitsandbytes

"$TOOLBIN/python" -m pip install -U tensorboard tensorboardX

export PATH="/root/.local/share/uv/tools/llamafactory/bin:$PATH"

command -v torchrun && torchrun --help | head -n 3

"$TOOLBIN/python" -m pip install "transformers==4.51.1"
```

Run:
```bash
bash env_llama_factory.sh
```

---

## Quick Start

- Enable BFT in your training config:
```
# File: path/to/your_bft_config.yaml

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all
deepspeed: /xxx/LLaMA-Factory/examples/deepspeed/ds_z3_config.json  # choices: [ds_z0_config.json, ds_z2_config.json, ds_z3_config.json]
use_bft_loss: true
```

- Typical launch:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llamafactory-cli train path/to/your_bft_config.yaml
```

- Optional environment variables for BFT:
  - BFT_GROUP_SIZE (default 256): sliding window length for group confidence

---

## Method

This section explains how BFT relates to SFT and DFT and then shows minimal code snippets that implement the idea.

### Supervised Fine-Tuning (SFT)

SFT is standard supervised training over instruction–response pairs. It minimizes token-level cross-entropy between the model’s predicted distribution and the ground-truth tokens. In practice, this objective implicitly puts very large weights on tokens that the model currently assigns low probability to, which can cause instability (e.g., gradient spikes) early in training.

### Dynamic Fine-Tuning (DFT)

DFT stabilizes SFT by down-weighting “easy” tokens and reducing the amplification for “hard” tokens. Concretely, it multiplies each token’s cross-entropy by the model’s own target probability (with stop-gradient), so training is less sensitive to low-probability labels. While DFT greatly improves stability, it can still under-train difficult samples because it operates only at the token level.

Minimal implementation (token-level reweighting):
```python
# dft_loss_func: entry point
def dft_loss_func(outputs, labels, num_items_in_batch=None):
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0))

    logits = logits.float()
    vocab_size = logits.size(-1)

    # shift alignment
    labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels[..., 1:].contiguous()

    # flatten
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1).to(logits.device)

    loss = _dft_cross_entropy(logits, shift_labels, num_items_in_batch)
    return loss


# _dft_cross_entropy: CE reweighted by detached target probability
def _dft_cross_entropy(source, target, num_items_in_batch=None, ignore_index=-100):
    per_token_loss = torch.nn.functional.cross_entropy(
        source, target, ignore_index=ignore_index, reduction="none"
    )
    valid_mask = (target != ignore_index)
    if not valid_mask.any():
        return torch.tensor(0.0, device=source.device, dtype=source.dtype)

    valid_losses = per_token_loss[valid_mask]
    with torch.no_grad():
        target_probs = torch.exp(-valid_losses)  # detached token weights

    weighted_losses = valid_losses * target_probs
    if num_items_in_batch is not None:
        total_loss = weighted_losses.sum()
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(total_loss.device)
        return total_loss / num_items_in_batch
    else:
        return weighted_losses.mean()
```

### Balanced Fine-Tuning (BFT)

BFT adds a sample-level balance on top of DFT to focus training on underconfident examples while preserving DFT’s token-level stability.

- Per-token confidence: the model’s probability assigned to the ground-truth token at each position.
- Group confidence: average per-token confidence within a sliding window (length g, stride 1) along the sequence.
- Lowest group confidence: the minimum group confidence across all windows; this summarizes the weakest span in the sample.
- Sample-level weight: s_b = 1 − p_conf, where p_conf is the lowest group confidence. Intuition: confident samples get smaller weights; underconfident samples get larger weights.
- Final BFT loss: compute per-sample DFT loss first, then multiply by s_b, and average over the batch.

Minimal implementation (sample-level confidence × token-level DFT):
```python
# lowest group confidence: [0, 1]
def _lowest_group_confidence(logits, labels, ignore_index=-100, group_size=None):
    import os
    if group_size is None:
        group_size = int(os.getenv("BFT_GROUP_SIZE", "256"))

    with torch.no_grad():
        B, T, V = logits.shape
        logits = logits.float()

        # shift alignment for labels and valid mask
        labels_pad = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels_pad[..., 1:].contiguous()[:, :T]
        valid_mask = (shift_labels != ignore_index)

        # log-probs of target tokens
        log_probs = logits.log_softmax(dim=-1)
        gather_idx = shift_labels.masked_fill(~valid_mask, 0).unsqueeze(-1)
        per_tok_logp = torch.gather(log_probs, dim=-1, index=gather_idx).squeeze(-1)

        p_conf_list = []
        for b in range(B):
            v = per_tok_logp[b][valid_mask[b]]
            if v.numel() == 0:
                p_conf_list.append(torch.tensor(0.0, device=logits.device))
                continue

            if v.numel() < group_size:
                min_avg_logp = v.mean()
            else:
                x = v.view(1, 1, -1)
                w = torch.ones(1, 1, group_size, device=logits.device) / group_size
                y = torch.nn.functional.conv1d(x, w, stride=1)  # sliding avg
                min_avg_logp = y.min().squeeze()

            p_conf = torch.exp(min_avg_logp).clamp(0.0, 1.0)
            p_conf_list.append(p_conf)

        return torch.stack(p_conf_list)  # shape: [B]


# bft_loss_func: sample-confidence-weighted DFT
def bft_loss_func(outputs, labels, num_items_in_batch=None):
    import os
    logits = outputs.get("logits")
    if logits is None:
        return outputs.get("loss", torch.tensor(0.0, device=labels.device))

    # sample confidence
    p_conf = _lowest_group_confidence(logits, labels, ignore_index=-100)
    gamma = float(os.getenv("BFT_CONF_GAMMA", "1.0"))
    p_conf = (p_conf ** gamma).detach()  # [B]

    # per-token CE
    B, T, V = logits.shape
    logits_flat = logits.float().view(-1, V)

    labels_pad = torch.nn.functional.pad(labels, (0, 1), value=-100)
    shift_labels = labels_pad[..., 1:].contiguous()[:, :T]        # [B, T]
    shift_labels_flat = shift_labels.reshape(-1)

    per_token_loss_flat = torch.nn.functional.cross_entropy(
        logits_flat, shift_labels_flat, ignore_index=-100, reduction="none"
    )
    per_token_loss = per_token_loss_flat.view(B, T)               # [B, T]

    # DFT token weights (detached)
    with torch.no_grad():
        dft_weights = torch.exp(-per_token_loss)                  # [B, T]

    # per-sample DFT
    weighted_loss = per_token_loss * dft_weights                  # [B, T]
    valid_mask = (shift_labels != -100).float()                   # [B, T]
    sample_dft_loss = (weighted_loss * valid_mask).sum(dim=1) / (valid_mask.sum(dim=1) + 1e-8)

    # sample-level scaling: (1 - p_conf)
    sample_weight = (1.0 - p_conf).to(sample_dft_loss.dtype)      # [B]
    final_loss = (sample_weight * sample_dft_loss).mean()
    return final_loss
```

---

## How to Enable BFT/DFT (Arguments and Trainer Hook)

- New training flags in finetuning arguments:
```python
    use_dft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the DFT loss."},
    )
    use_bft_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use the BFT loss (dynamic scheduling between SFT and DFT)."},
    )
```

- Hooked into the SFT trainer:
```python
        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func
            self.compute_loss_func = dft_loss_func

        if finetuning_args.use_bft_loss:
            from ..trainer_utils import bft_loss_func
            self.compute_loss_func = bft_loss_func
```


## Related Repositories

- [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [yongliang-wu/DFT](https://github.com/yongliang-wu/DFT)
- [facebookresearch/deepconf](https://github.com/facebookresearch/deepconf)