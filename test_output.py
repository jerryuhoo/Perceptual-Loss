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
    nfilts = 64
    mT_dB_shift = 20
    normalize = False
    if normalize:
        ref_dB = 120

    waveform, sample_rate = torchaudio.load("test_wavs/v_gt.wav")
    audio_gt = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_good.wav")
    audio_good = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_bad.wav")
    assert fs == sample_rate
    audio_bad = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_gt_quiet.wav")
    audio_gt_quiet = waveform[0]

    waveform, sample_rate = torchaudio.load("test_wavs/v_gt_rev_quiet.wav")
    audio_gt_rev_quiet = waveform[0]

    # Compute STFT
    ys_gt = compute_STFT(audio_gt, N=N, normalize=normalize).unsqueeze(0).unsqueeze(0)
    ys_good = (
        compute_STFT(audio_good, N=N, normalize=normalize).unsqueeze(0).unsqueeze(0)
    )
    ys_bad = compute_STFT(audio_bad, N=N, normalize=normalize).unsqueeze(0).unsqueeze(0)
    ys_gt_quiet = (
        compute_STFT(audio_gt_quiet, N=N, normalize=normalize).unsqueeze(0).unsqueeze(0)
    )
    ys_gt_rev_quiet = (
        compute_STFT(audio_gt_rev_quiet, N=N, normalize=normalize)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    mse_loss_good_gt = F.mse_loss(ys_gt, ys_good)
    print("mse_loss_good_gt", mse_loss_good_gt.item())  # loss should be small
    mse_loss_bad_gt = F.mse_loss(ys_gt, ys_bad)
    print("mse_loss_bad_gt", mse_loss_bad_gt.item())  # loss should be large
    print(
        "mse loss: good/bad ratio, small is better",
        mse_loss_good_gt.item() / mse_loss_bad_gt.item(),
    )
    print("====================================================")

    # single file example with weighting
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=False,
        mT_shift=0.00,
    )
    print("weighted loss: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=False,
        mT_shift=0.00,
    )
    print("weighted loss: bad, gt", ploss_bad.item())
    print(
        "weighted loss: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )
    print("====================================================")

    # single file example without weighting
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=False,
    )
    print("psy loss: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
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
    print("====================================================")

    # single file example with weighting and LTQ
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
    )
    print("weighted loss + LTQ: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
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
    print("====================================================")

    # single file example without weighting and with LTQ
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=True,
    )
    print("psy loss + LTQ: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
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
    print("====================================================")

    # single file example with weighting and with LTQ, with mT_dB_shift
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=mT_dB_shift,
        use_dB=False,
    )
    print(f"weighted loss + LTQ + shift {mT_dB_shift}: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=mT_dB_shift,
        use_dB=False,
    )
    print(f"weighted loss + LTQ + shift {mT_dB_shift}: bad, gt", ploss_bad.item())
    print(
        f"weighted loss + LTQ + shift {mT_dB_shift}: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )
    print("====================================================")

    # single file example with weighting and with LTQ, with mT_dB_shift, with dB scale
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=mT_dB_shift,
        use_dB=True,
    )
    print(
        f"weighted loss + LTQ + shift {mT_dB_shift} + dB: good, gt", ploss_good.item()
    )

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=mT_dB_shift,
        use_dB=True,
    )
    print(f"weighted loss + LTQ + shift {mT_dB_shift} + dB: bad, gt", ploss_bad.item())
    print(
        f"weighted loss + LTQ + shift {mT_dB_shift} + dB: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )
    print("====================================================")

    # single file example with weighting and with LTQ, with dB scale
    ploss_good = psycho_acoustic_loss(
        ys_good,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=0,
        use_dB=True,
    )
    print(f"weighted loss + LTQ + dB: good, gt", ploss_good.item())

    ploss_bad = psycho_acoustic_loss(
        ys_bad,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        mT_shift=0,
        use_dB=True,
    )
    print(f"weighted loss + LTQ + dB: bad, gt", ploss_bad.item())
    print(
        "weighted loss + LTQ + dB: good/bad ratio, small is better",
        ploss_good.item() / ploss_bad.item(),
    )
    print("====================================================")

    mse_loss_quiet = F.mse_loss(ys_gt_quiet, ys_gt_rev_quiet)
    print("ys_gt_quiet", ys_gt_quiet.shape)
    print("ys_gt_rev_quiet", ys_gt_rev_quiet.shape)
    print("ys_gt", ys_gt.shape)
    mse_loss_quiet_gt = F.mse_loss(ys_gt, ys_gt_quiet)
    print("mse_loss_quiet", mse_loss_quiet.item())
    print("mse_loss_quiet_gt", mse_loss_quiet_gt.item())
    print(
        "mse loss: quiet/gt ratio, small is better",
        mse_loss_quiet.item() / mse_loss_quiet_gt.item(),
    )

    ploss_quiet = psycho_acoustic_loss(
        ys_gt_rev_quiet,
        ys_gt_quiet,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        use_dB=False,
    )
    ploss_quiet_gt = psycho_acoustic_loss(
        ys_gt_quiet,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        use_dB=False,
    )
    print("psy loss: quiet, rev_quiet", ploss_quiet.item())
    print("psy loss: quiet_gt, gt", ploss_quiet_gt.item())
    print(
        "psy loss: quiet/quiet_gt ratio, small is better",
        ploss_quiet.item() / ploss_quiet_gt.item(),
    )

    ploss_quiet = psycho_acoustic_loss(
        ys_gt_rev_quiet,
        ys_gt_quiet,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=True,
        use_dB=True,
    )
    ploss_quiet_gt = psycho_acoustic_loss(
        ys_gt_quiet,
        ys_gt,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=False,
        use_LTQ=True,
        use_dB=True,
    )
    print("psy loss without LTQ: quiet, rev_quiet", ploss_quiet.item())
    print("psy loss without LTQ: quiet_gt, gt", ploss_quiet_gt.item())
    print(
        "psy loss without LTQ: quiet/quiet_gt ratio, small is better",
        ploss_quiet.item() / ploss_quiet_gt.item(),
    )

    # only test
    ys_gt_noise = torch.rand(ys_gt.shape)
    ys_gt_sin = torch.sin(ys_gt)
    ploss = psycho_acoustic_loss(
        ys_gt_rev_quiet,
        ys_gt_noise,
        fs=sample_rate,
        N=N,
        nfilts=nfilts,
        use_weighting=True,
        use_LTQ=True,
        use_dB=True,
    )


if __name__ == "__main__":
    main()
