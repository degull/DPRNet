# train.py
# 원본
import os
import yaml
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

CONFIG_PATH = "configs/dpr_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def save_checkpoint(model, optimizer, epoch, loss, psnr, ssim, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"checkpoint_epoch_{epoch:02d}_loss_{loss:.4f}_psnr_{psnr:.2f}_ssim_{ssim:.4f}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "psnr": psnr,
            "ssim": ssim,
        },
        save_path,
    )
    log_print(f"\n💾 Checkpoint saved: {save_path}", log_file)


def main():
    config = load_config(CONFIG_PATH)

    os.makedirs(config["paths"]["log_dir"], exist_ok=True)
    log_file = os.path.join(config["paths"]["log_dir"], "training_log.txt")

    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_print(f"\n{'=' * 60}", log_file)
    log_print(f"🚀 TRAINING STARTED AT: {start_time_str}", log_file)
    log_print(f"{'=' * 60}", log_file)

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    # Dataset
    log_print("💿 Initializing Datasets...", log_file)
    dataset = DPRDataset(
        data_root=config["paths"]["root_dir"],
        metadata_file=config["paths"]["metadata_file"],
        tokenizer_path=config["model"]["llm_model_id"],
        max_length=config["model"]["max_length"],
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=DPRDataset.collate_fn,
        pin_memory=True,
    )
    log_print(f"   - Total Images: {len(dataset)}", log_file)

    # Model
    log_print("🏗️ Building DPR-Net V2 Model (4-bit)...", log_file)
    model = DPRNetV2(config).to(device)

    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=1e-4
    )

    criterion = nn.L1Loss()
    scaler = GradScaler()
    accum_steps = config["train"]["accumulate_grad_batches"]

    metric_psnr = PeakSignalNoiseRatio().to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    num_epochs = config["train"]["num_epochs"]

    # ============================================================
    # TRAIN LOOP
    # ============================================================
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        t0 = time.time()

        log_print(f"\n[Epoch {epoch}/{num_epochs}] Started...", log_file)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_raw = batch["attention_mask"].to(device)
            vet_input = batch["vet_input"].to(device)
            vet_target = batch["vet_target"].to(device)

            B, T = input_ids.shape
            attn_mask = torch.cat([torch.ones(B, 257, device=device), attn_raw], dim=1) \
                        if attn_raw.shape[1] == T else attn_raw

            with autocast(dtype=torch.float16):
                restored = model({
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "high_res_images": vet_input,
                })
                restored_clamped = torch.clamp(restored, 0, 1)
                loss = criterion(restored, vet_target)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                batch_psnr = metric_psnr(restored_clamped, vet_target).item()
                batch_ssim = metric_ssim(restored_clamped, vet_target).item()

                epoch_loss += loss.item()
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

            # 🔥 tqdm 진행바 postfix 추가 → 여기에서 뜬다
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{batch_psnr:.2f}",
                ssim=f"{batch_ssim:.4f}",
            )

            # 🔥 100 iteration 마다 콘솔 + training_log.txt 출력
            if (step % 100) == 0:
                iter_msg = (
                    f"[Epoch {epoch}] Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | PSNR: {batch_psnr:.2f} | SSIM: {batch_ssim:.4f}"
                )
                progress_bar.write(iter_msg)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(iter_msg + "\n")

        epoch_loss /= len(train_loader)
        epoch_psnr /= len(train_loader)
        epoch_ssim /= len(train_loader)
        elapsed = time.time() - t0

        log_print(
            f"\n📊 Epoch {epoch} Done — Time: {elapsed:.2f}s | "
            f"Loss: {epoch_loss:.5f} | PSNR: {epoch_psnr:.2f} | SSIM: {epoch_ssim:.4f}",
            log_file
        )

        save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_psnr, epoch_ssim,
                        config["paths"]["log_dir"], log_file)

    log_print("🏁 Training Finished Successfully!", log_file)


if __name__ == "__main__":
    main()


# 이어서 학습
# train.py
""" import os
import yaml
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from models.dpr_net_v2 import DPRNetV2
from data.dataset import DPRDataset

CONFIG_PATH = "configs/dpr_config.yaml"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_print(message, log_file):
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def find_last_checkpoint(save_dir):
    ckpts = [f for f in os.listdir(save_dir) if f.startswith("checkpoint_epoch_") and f.endswith(".pth")]
    if not ckpts:
        return None, 0
    ckpts_sorted = sorted(ckpts, key=lambda x: int(x.split("_")[2]))
    last_ckpt = ckpts_sorted[-1]
    last_epoch = int(last_ckpt.split("_")[2])
    return os.path.join(save_dir, last_ckpt), last_epoch


def save_checkpoint(model, optimizer, epoch, loss, psnr, ssim, save_dir, log_file):
    os.makedirs(save_dir, exist_ok=True)
    filename = f"checkpoint_epoch_{epoch:02d}_loss_{loss:.4f}_psnr_{psnr:.2f}_ssim_{ssim:.4f}.pth"
    save_path = os.path.join(save_dir, filename)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "psnr": psnr,
            "ssim": ssim,
        },
        save_path,
    )
    log_print(f"\n💾 Checkpoint saved: {save_path}", log_file)


def main():
    config = load_config(CONFIG_PATH)
    save_dir = config["paths"]["log_dir"]

    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, "training_log.txt")

    # 로그 이어쓰기 — 기존 로그 아래에 계속 기록됨
    log_print("\n" + "=" * 60, log_file)
    log_print(f"🚀 TRAINING STARTED AT: {datetime.datetime.now()}", log_file)
    log_print("=" * 60, log_file)

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    # Dataset
    dataset = DPRDataset(
        data_root=config["paths"]["root_dir"],
        metadata_file=config["paths"]["metadata_file"],
        tokenizer_path=config["model"]["llm_model_id"],
        max_length=config["model"]["max_length"],
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=config["train"]["num_workers"],
        collate_fn=DPRDataset.collate_fn,
        pin_memory=True,
    )
    log_print(f"💿 Total Images: {len(dataset)}", log_file)

    # Model
    model = DPRNetV2(config).to(device)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(config["train"]["learning_rate"]),
        weight_decay=1e-4
    )
    criterion = nn.L1Loss()
    scaler = GradScaler()
    accum_steps = config["train"]["accumulate_grad_batches"]

    metric_psnr = PeakSignalNoiseRatio().to(device)
    metric_ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    # ==============================
    # 🔥 Resume from latest checkpoint
    # ==============================
    ckpt_path, last_epoch = find_last_checkpoint(save_dir)
    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = last_epoch + 1
        log_print(f"🔁 Resuming from: {ckpt_path} (start at epoch {start_epoch})", log_file)
    else:
        start_epoch = 1
        log_print("📌 No previous checkpoint found — starting from epoch 1", log_file)

    num_epochs = config["train"]["num_epochs"]

    # ==============================
    # TRAIN LOOP
    # ==============================
    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        t0 = time.time()

        log_print(f"\n[Epoch {epoch}/{num_epochs}] Started...", log_file)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=True)

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress_bar, start=1):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attn_raw = batch["attention_mask"].to(device)
            vet_input = batch["vet_input"].to(device)
            vet_target = batch["vet_target"].to(device)

            B, T = input_ids.shape
            attn_mask = torch.cat([torch.ones(B, 257, device=device), attn_raw], dim=1) \
                        if attn_raw.shape[1] == T else attn_raw

            with autocast(dtype=torch.float16):
                restored = model({
                    "pixel_values": pixel_values,
                    "input_ids": input_ids,
                    "attention_mask": attn_mask,
                    "high_res_images": vet_input,
                })
                restored_clamped = torch.clamp(restored, 0, 1)
                loss = criterion(restored, vet_target)
                loss_scaled = loss / accum_steps

            scaler.scale(loss_scaled).backward()

            if step % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                batch_psnr = metric_psnr(restored_clamped, vet_target).item()
                batch_ssim = metric_ssim(restored_clamped, vet_target).item()
                epoch_loss += loss.item()
                epoch_psnr += batch_psnr
                epoch_ssim += batch_ssim

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                psnr=f"{batch_psnr:.2f}",
                ssim=f"{batch_ssim:.4f}",
            )

            if step % 100 == 0:
                msg = (
                    f"[Epoch {epoch}] Step {step}/{len(train_loader)} | "
                    f"Loss: {loss.item():.4f} | PSNR: {batch_psnr:.2f} | SSIM: {batch_ssim:.4f}"
                )
                progress_bar.write(msg)
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

        epoch_loss /= len(train_loader)
        epoch_psnr /= len(train_loader)
        epoch_ssim /= len(train_loader)
        elapsed = time.time() - t0

        log_print(
            f"\n📊 Epoch {epoch} Done — Time: {elapsed:.2f}s | "
            f"Loss: {epoch_loss:.5f} | PSNR: {epoch_psnr:.2f} | SSIM: {epoch_ssim:.4f}",
            log_file
        )

        save_checkpoint(model, optimizer, epoch, epoch_loss, epoch_psnr, epoch_ssim, save_dir, log_file)

    log_print("\n🏁 Training Finished Successfully!", log_file)


if __name__ == "__main__":
    main()
 """