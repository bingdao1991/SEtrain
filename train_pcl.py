import os
import torch
import random
import shutil
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm
from glob import glob
from pesq import pesq
from joblib import Parallel, delayed
import soundfile as sf
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from distributed_utils import reduce_value
from models.ulunas_pcl import ULUNAS_PCL
from models.pcl_modules import PatchSampleF, PatchNCELoss, PCLFeatureExtractor
from loss_factory import DualOutputLoss, LossWeightScheduler
from dataloader import DNS3DualOutputDataset as Dataset
from scheduler import LinearWarmupCosineAnnealingLR as WarmupLR
from metrics import evaluate_dual_output_batch

seed = 43
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_pretrained_model(config, device):
    model = ULUNAS_PCL(**config["network_config"]).to(device)
    pretrain_cfg = config.get("pretrain", {})
    if not pretrain_cfg.get("enabled", False):
        return model

    ckpt_path = pretrain_cfg.get("ulunas_checkpoint")
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("Pretrain enabled but checkpoint not found, training from scratch.")
        return model

    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = ULUNAS_PCL.from_ulunas_checkpoint(
        state_dict,
        init_noise_mask=pretrain_cfg.get("init_noise_mask", "complement"),
        **config["network_config"],
    ).to(device)
    print(f"Loaded ULUNAS pretrained weights from {ckpt_path}")
    return model


def build_pcl_modules(config, device, batch_size):
    pcl_cfg = config.get("pcl", {})
    in_channels = config["network_config"].get("channels", [12, 24, 24, 32, 16])[-1]
    patch_sampler = PatchSampleF(
        in_channels=in_channels,
        embed_dim=pcl_cfg.get("embed_dim", 128),
        use_mlp=pcl_cfg.get("use_mlp", True),
    ).to(device)
    pcl_extractor = PCLFeatureExtractor(
        patch_sampler=patch_sampler,
        num_patches=pcl_cfg.get("num_patches", 256),
    ).to(device)
    pcl_loss = PatchNCELoss(
        temperature=pcl_cfg.get("temperature", 0.07),
        batch_size=batch_size,
    ).to(device)
    return pcl_extractor, pcl_loss


def run(rank, config, args):
    if args.world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        torch.cuda.set_device(rank)
        dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)
    batch_size = config["train_dataloader"]["batch_size"]

    collate_fn = Dataset.collate_fn
    shuffle = False if args.world_size > 1 else True

    train_dataset = Dataset(**config["train_dataset"])
    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.world_size > 1
        else None
    )
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        sampler=train_sampler,
        collate_fn=collate_fn,
        shuffle=shuffle,
        **config["train_dataloader"],
    )

    validation_dataset = Dataset(**config["validation_dataset"])
    validation_sampler = (
        torch.utils.data.distributed.DistributedSampler(validation_dataset)
        if args.world_size > 1
        else None
    )
    validation_dataloader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        sampler=validation_sampler,
        collate_fn=collate_fn,
        shuffle=False,
        **config["validation_dataloader"],
    )

    model = load_pretrained_model(config, args.device)
    pcl_extractor, pcl_loss = build_pcl_modules(config, args.device, batch_size)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    params = list(model.parameters()) + list(pcl_extractor.parameters())
    optimizer = torch.optim.Adam(params=params, **config["optimizer"])
    scheduler = WarmupLR(optimizer, **config["scheduler"]["kwargs"])

    loss_cfg = dict(config["loss"])
    loss_cfg["pcl_loss"] = pcl_loss
    loss_func = DualOutputLoss(**loss_cfg).to(args.device)

    loss_weight_scheduler = LossWeightScheduler(
        lambda_speech=loss_cfg.get("lambda_speech", 1.0),
        lambda_noise=loss_cfg.get("lambda_noise", 0.5),
        lambda_recon=loss_cfg.get("lambda_recon", 0.3),
        lambda_mask_sum=loss_cfg.get("lambda_mask_sum", 0.0),
        lambda_pcl_target=config.get("pcl", {}).get("lambda_pcl_target", 2.0),
        pcl_warmup_steps=config.get("pcl", {}).get("warmup_steps", 10000),
        training_phase=config.get("training_phase", "phase2"),
    )

    trainer = TrainerPCL(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_func=loss_func,
        pcl_extractor=pcl_extractor,
        loss_weight_scheduler=loss_weight_scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        train_sampler=train_sampler,
        args=args,
    )
    trainer.train()

    if args.world_size > 1:
        dist.destroy_process_group()


class TrainerPCL:
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        loss_func,
        pcl_extractor,
        loss_weight_scheduler,
        train_dataloader,
        validation_dataloader,
        train_sampler,
        args,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_func = loss_func
        self.pcl_extractor = pcl_extractor
        self.loss_weight_scheduler = loss_weight_scheduler
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.train_sampler = train_sampler
        self.rank = args.rank
        self.device = args.device
        self.world_size = args.world_size
        self.batch_size = config["train_dataloader"]["batch_size"]
        self.global_step = 0

        config["DDP"]["world_size"] = args.world_size
        self.trainer_config = config["trainer"]
        self.epochs = self.trainer_config["epochs"]
        self.save_checkpoint_interval = self.trainer_config["save_checkpoint_interval"]
        self.clip_grad_norm_value = self.trainer_config["clip_grad_norm_value"]
        self.resume = self.trainer_config["resume"]
        self.use_validation = self.trainer_config.get("use_validation", False)

        if not self.resume:
            self.exp_path = (
                self.trainer_config["exp_path"]
                + "_pcl_"
                + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
            )
        else:
            self.exp_path = (
                self.trainer_config["exp_path"]
                + "_pcl_"
                + self.trainer_config["resume_datetime"]
            )

        self.log_path = os.path.join(self.exp_path, "logs")
        self.checkpoint_path = os.path.join(self.exp_path, "checkpoints")
        self.sample_path = os.path.join(self.exp_path, "val_samples")
        self.code_path = os.path.join(self.exp_path, "codes")

        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)
        os.makedirs(self.code_path, exist_ok=True)

        if self.rank == 0:
            OmegaConf.save(OmegaConf.create(config), os.path.join(self.exp_path, "config.yaml"))
            shutil.copy2(__file__, self.exp_path)
            for file in Path(__file__).parent.iterdir():
                if file.is_file():
                    shutil.copy2(file, self.code_path)
            shutil.copytree(
                Path(__file__).parent / "models",
                Path(self.code_path) / "models",
                dirs_exist_ok=True,
            )
            self.writer = SummaryWriter(self.log_path)

        self.start_epoch = 1
        self.best_score = 0
        self.state_dict_best = None

        if self.resume:
            self._resume_checkpoint()

    def _unwrap_model(self):
        return self.model.module if self.world_size > 1 else self.model

    def train_step(self, noisy, clean, noise_gt):
        weights = self.loss_weight_scheduler.get_weights(self.global_step)
        self.loss_func.set_lambda_pcl(weights["lambda_pcl"])

        out = self._unwrap_model()(noisy, return_dict=True)

        pcl_features = None
        if weights["lambda_pcl"] > 0 and self.config.get("pcl", {}).get("enabled", False):
            with torch.no_grad():
                latent_gt = self._unwrap_model().encode_latent(clean)
            pcl_features = self.pcl_extractor(
                out.latent_speech, out.latent_noise, latent_gt
            )

        loss_out = self.loss_func(
            pred_speech=out.speech,
            pred_noise=out.noise,
            gt_speech=clean,
            gt_noise=noise_gt,
            mixture=noisy,
            mask_speech=out.mask_speech,
            mask_noise=out.mask_noise,
            pcl_features=pcl_features,
            batch_size=noisy.shape[0],
            return_details=True,
        )
        return loss_out

    def _save_checkpoint(self, epoch, score):
        model_dict = (
            self.model.module.state_dict()
            if self.world_size > 1
            else self.model.state_dict()
        )
        state_dict = {
            "epoch": epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "model": model_dict,
            "pcl_extractor": self.pcl_extractor.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(
            state_dict,
            os.path.join(self.checkpoint_path, f"model_{str(epoch).zfill(3)}.tar"),
        )
        if score >= self.best_score or self.state_dict_best is None:
            self.state_dict_best = state_dict.copy()
            self.best_score = max(self.best_score, score)

    def _resume_checkpoint(self):
        latest_checkpoints = sorted(glob(os.path.join(self.checkpoint_path, "model_*.tar")))[-1]
        checkpoint = torch.load(latest_checkpoints, map_location=self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint.get("global_step", 0)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.world_size > 1:
            self.model.module.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint["model"])
        if "pcl_extractor" in checkpoint:
            self.pcl_extractor.load_state_dict(checkpoint["pcl_extractor"])

    def _train_epoch(self, epoch):
        total_loss = 0.0
        if hasattr(self.train_dataloader.dataset, "sample_data_per_epoch"):
            self.train_dataloader.dataset.sample_data_per_epoch()
        self.train_bar = tqdm(self.train_dataloader, ncols=120)

        for step, (noisy, clean, noise_gt) in enumerate(self.train_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            noise_gt = noise_gt.to(self.device)

            loss_out = self.train_step(noisy, clean, noise_gt)
            loss = loss_out.total
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()
            self.global_step += 1

            self.train_bar.desc = "train[{}/{}][{}]".format(
                epoch,
                self.epochs + self.start_epoch - 1,
                datetime.now().strftime("%Y-%m-%d-%H:%M"),
            )
            self.train_bar.postfix = "loss={:.3f}, sp={:.3f}, ns={:.3f}, pcl={:.3f}".format(
                total_loss / step,
                loss_out.details["speech"],
                loss_out.details["noise"],
                loss_out.details["pcl"],
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.pcl_extractor.parameters()),
                self.clip_grad_norm_value,
            )
            self.optimizer.step()

            if self.config["scheduler"]["update_interval"] == "step":
                self.scheduler.step()

            if self.rank == 0 and step % 100 == 0:
                self.writer.add_scalars(
                    "train_step_loss",
                    {
                        "total": loss_out.details["speech"]
                        + loss_out.details["noise"]
                        + loss_out.details["recon"]
                        + loss_out.details["pcl"],
                        "speech": loss_out.details["speech"],
                        "noise": loss_out.details["noise"],
                        "recon": loss_out.details["recon"],
                        "pcl": loss_out.details["pcl"],
                    },
                    self.global_step,
                )

        if self.world_size > 1 and self.device != torch.device("cpu"):
            torch.cuda.synchronize(self.device)

        if self.rank == 0:
            self.writer.add_scalars("lr", {"lr": self.optimizer.param_groups[0]["lr"]}, epoch)
            self.writer.add_scalars("train_loss", {"train_loss": total_loss / step}, epoch)

    @torch.inference_mode()
    def _validation_epoch(self, epoch):
        total_loss = 0.0
        total_pesq = 0.0
        total_metrics = {"speech_si_snr": 0.0, "noise_si_snr": 0.0, "recon_si_snr": 0.0}

        self.validation_bar = tqdm(self.validation_dataloader, ncols=130)
        for step, (noisy, clean, noise_gt) in enumerate(self.validation_bar, 1):
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            noise_gt = noise_gt.to(self.device)

            out = self._unwrap_model()(noisy, return_dict=True)
            loss_out = self.loss_func(
                pred_speech=out.speech,
                pred_noise=out.noise,
                gt_speech=clean,
                gt_noise=noise_gt,
                mixture=noisy,
                return_details=True,
            )
            loss = loss_out.total
            if self.world_size > 1:
                loss = reduce_value(loss)
            total_loss += loss.item()

            batch_metrics = evaluate_dual_output_batch(
                out.speech, out.noise, clean, noise_gt, noisy
            )
            for key in total_metrics:
                total_metrics[key] += batch_metrics[key]

            clean_np = clean.cpu().numpy()
            speech_np = out.speech.detach().cpu().numpy()
            pesq_score_batch = Parallel(n_jobs=1)(
                delayed(pesq)(16000, c, e, "wb") for c, e in zip(clean_np, speech_np)
            )
            pesq_score = torch.tensor(pesq_score_batch, device=self.device).mean()
            if self.world_size > 1:
                pesq_score = reduce_value(pesq_score)
            total_pesq += pesq_score.item()

            if self.rank == 0 and (epoch == 1 or epoch % 10 == 0) and step <= 3:
                noisy_np = noisy.cpu().numpy()
                noise_np = out.noise.detach().cpu().numpy()
                base = os.path.join(self.sample_path, f"sample_{step}")
                if not os.path.exists(base + "_noisy.wav"):
                    sf.write(base + "_noisy.wav", noisy_np[0], self.config["samplerate"])
                    sf.write(base + "_clean.wav", clean_np[0], self.config["samplerate"])
                sf.write(
                    base + f"_speech_epoch{str(epoch).zfill(3)}.wav",
                    speech_np[0],
                    self.config["samplerate"],
                )
                sf.write(
                    base + f"_noise_epoch{str(epoch).zfill(3)}.wav",
                    noise_np[0],
                    self.config["samplerate"],
                )

            self.validation_bar.postfix = (
                f"loss={total_loss / step:.3f}, pesq={total_pesq / step:.3f}, "
                f"sp_snr={total_metrics['speech_si_snr'] / step:.2f}"
            )

        if self.rank == 0:
            self.writer.add_scalars(
                "val_metrics",
                {
                    "loss": total_loss / step,
                    "pesq": total_pesq / step,
                    **{k: v / step for k, v in total_metrics.items()},
                },
                epoch,
            )
        return total_loss / step, total_pesq / step

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            self.model.train()
            self.pcl_extractor.train()
            self._train_epoch(epoch)

            score = 0.0
            if self.use_validation:
                self.model.eval()
                self.pcl_extractor.eval()
                _, score = self._validation_epoch(epoch)

            if self.config["scheduler"]["update_interval"] == "epoch":
                self.scheduler.step()

            if self.rank == 0 and epoch % self.save_checkpoint_interval == 0:
                self._save_checkpoint(epoch, score)

        if self.rank == 0 and self.state_dict_best is not None:
            torch.save(
                self.state_dict_best,
                os.path.join(
                    self.checkpoint_path,
                    "best_model_{}.tar".format(
                        str(self.state_dict_best["epoch"]).zfill(3)
                    ),
                ),
            )
            print("------------Training done!------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_train_pcl.yaml")
    parser.add_argument("-D", "--device", default="0", help="GPU index, e.g. 0 or 0,1")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    args.world_size = len(args.device.split(","))
    config = OmegaConf.load(args.config)

    if args.world_size > 1:
        torch.multiprocessing.spawn(run, args=(config, args), nprocs=args.world_size, join=True)
    else:
        run(0, config, args)
