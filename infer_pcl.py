import os
import torch
import soundfile as sf
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from models.ulunas_pcl import ULUNAS_PCL


def load_model(checkpoint_path, network_config, device):
    model = ULUNAS_PCL(**network_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def infer_file(model, wav_path, device):
    noisy, fs = sf.read(wav_path, dtype="float32")
    if noisy.ndim > 1:
        noisy = noisy.mean(axis=-1)
    inp = torch.FloatTensor(noisy).unsqueeze(0).to(device)
    with torch.inference_mode():
        speech, noise = model(inp)
    return speech.cpu().numpy().squeeze(), noise.cpu().numpy().squeeze(), fs


def infer_directory(cfg_infer):
    cfg_network = OmegaConf.load(cfg_infer.network.config)
    network_config = cfg_network.get("network_config", cfg_network)

    device = torch.device(
        f"cuda:{cfg_infer.device}" if torch.cuda.is_available() else "cpu"
    )
    model = load_model(cfg_infer.network.checkpoint, network_config, device)

    noisy_folder = cfg_infer.test_dataset.noisy_dir
    speech_out_dir = cfg_infer.network.speech_out_dir
    noise_out_dir = cfg_infer.network.noise_out_dir
    os.makedirs(speech_out_dir, exist_ok=True)
    os.makedirs(noise_out_dir, exist_ok=True)

    noisy_wavs = sorted([x for x in os.listdir(noisy_folder) if x.endswith(".wav")])

    speech_scp, noise_scp, ref_scp = [], [], []
    clean_folder = cfg_infer.test_dataset.get("clean_dir")

    for wav_name in tqdm(noisy_wavs, desc="infer_pcl"):
        wav_path = os.path.join(noisy_folder, wav_name)
        speech, noise, fs = infer_file(model, wav_path, device)

        uid = wav_name.replace(".wav", "")
        speech_path = os.path.join(speech_out_dir, uid + "_speech.wav")
        noise_path = os.path.join(noise_out_dir, uid + "_noise.wav")
        sf.write(speech_path, speech, fs)
        sf.write(noise_path, noise, fs)

        speech_scp.append([uid, speech_path])
        noise_scp.append([uid, noise_path])
        if clean_folder is not None:
            ref_scp.append([uid, os.path.join(clean_folder, wav_name)])

    with open(os.path.join(speech_out_dir, "inf_speech.scp"), "w") as f:
        for uid, path in speech_scp:
            f.write(f"{uid} {path}\n")
    with open(os.path.join(noise_out_dir, "inf_noise.scp"), "w") as f:
        for uid, path in noise_scp:
            f.write(f"{uid} {path}\n")
    if ref_scp:
        with open(os.path.join(speech_out_dir, "ref.scp"), "w") as f:
            for uid, path in ref_scp:
                f.write(f"{uid} {path}\n")


def main(args):
    cfg_infer = OmegaConf.load(args.config)
    cfg_infer.device = args.device
    infer_directory(cfg_infer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-C", "--config", default="configs/cfg_infer_pcl.yaml")
    parser.add_argument("-D", "--device", default="0")
    args = parser.parse_args()
    main(args)
