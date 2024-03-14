import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from psycho_acoustic_loss import psycho_acoustic_loss, compute_STFT


def main():
    # Load audio
    fs = 44100
    N = 1024
    nfilts = 1024
    mT_dB_shift = 0.02
    waveform, sample_rate = torchaudio.load("test_wavs/v_gt.wav")
    audio_gt = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_good.wav")
    audio_good = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_bad.wav")
    assert fs == sample_rate
    audio_bad = waveform[0]

    # Compute STFT
    ys_gt_align = compute_STFT(audio_gt, N=N).unsqueeze(0).unsqueeze(0)
    ys_good = compute_STFT(audio_good, N=N).unsqueeze(0).unsqueeze(0)
    ys_bad = compute_STFT(audio_bad, N=N).unsqueeze(0).unsqueeze(0)

    ys_gt_align = ys_gt_align[:, :, :, : ys_good.shape[-1]]

    mse_loss_good_gt = F.mse_loss(ys_gt_align, ys_good)
    print("mse_loss_good_gt", mse_loss_good_gt.item())  # loss should be small
    mse_loss_bad_gt = F.mse_loss(ys_gt_align, ys_bad)
    print("mse_loss_bad_gt", mse_loss_bad_gt.item())  # loss should be large
    print(
        "mse loss: good/bad ratio, small is better",
        mse_loss_good_gt.item() / mse_loss_bad_gt.item(),
    )

    # single file example with weighting
    ploss_good = psycho_acoustic_loss(
        ys_gt_align,
        ys_good,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=False,
        mt_shift=0.00,
    )
    print("weighted loss: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_gt_align,
        ys_bad,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=False,
        mt_shift=0.00,
    )
    print("weighted loss: bad, gt", ploss_bad.item())
    print(
        "weighted loss: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )
    print("compare psyloss with mse, psyloss ratio should be smaller")

    # single file example without weighting
    ploss_good = psycho_acoustic_loss(
        ys_gt_align,
        ys_good,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=False,
    )
    print("psy loss: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_gt_align,
        ys_bad,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=False,
    )
    print("psy loss: bad, gt", ploss_bad.item())
    print(
        "psy loss: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )

    # single file example with weighting and LTQ
    ploss_good = psycho_acoustic_loss(
        ys_gt_align,
        ys_good,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
    )
    print("weighted loss + LTQ: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_gt_align,
        ys_bad,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
    )
    print("weighted loss + LTQ: bad, gt", ploss_bad.item())
    print(
        "weighted loss + LTQ: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )

    # single file example without weighting and with LTQ
    ploss_good = psycho_acoustic_loss(
        ys_gt_align,
        ys_good,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=True,
    )
    print("psy loss + LTQ: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_gt_align,
        ys_bad,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=True,
    )
    print("psy loss + LTQ: bad, gt", ploss_bad.item())
    print(
        "psy loss + LTQ: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )

    # single file example with weighting and with LTQ, with mT_dB_shift
    ploss_good = psycho_acoustic_loss(
        ys_gt_align,
        ys_good,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mt_shift=mT_dB_shift,
    )
    print(f"weighted loss + LTQ + shift {mT_dB_shift}: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_gt_align,
        ys_bad,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mt_shift=mT_dB_shift,
    )
    print(f"weighted loss + LTQ + shift {mT_dB_shift}: bad, gt", ploss_bad.item())
    print(
        "weighted loss + LTQ: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )


if __name__ == "__main__":
    main()
