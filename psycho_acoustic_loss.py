import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os


def psycho_acoustic_loss(
    ys_pred,
    ys_true,
    fs=44100,
    N=1024,
    nfilts=64,
    use_weighting=True,
    use_LTQ=False,
    mT_shift=0,
    ref_dB=60,
    use_dB=False,
):
    """
    ys_pred: [batch_size, channels, N+1, frame]
    ys_true: [batch_size, channels, N+1, frame]
    """
    # Check the number of channels (either 1 for mono or 2 for stereo)
    channels = ys_pred.shape[1]

    if channels not in [1, 2]:
        raise ValueError(
            f"Unsupported number of channels: {channels}, only mono and stereo are supported"
        )

    # Function to compute MSE loss for a single channel
    def compute_channel_loss(
        ys_pred, ys_true, use_weighting, use_LTQ, mT_shift, ref_dB=60
    ):
        mT_true = compute_masking_threshold(
            ys_true, fs, N, nfilts, use_LTQ=use_LTQ, ref_dB=ref_dB
        )
        if use_weighting:
            mT_true = mT_true.unsqueeze(1)
            W = mapping2barkmat(fs, nfilts, 2 * N).to(ys_pred.device)
            W_inv = mappingfrombarkmat(W, 2 * N).to(ys_pred.device)
            mT_true = mappingfrombark(mT_true, W_inv, 2 * N).transpose(-1, -2)

            if use_dB:
                mT_true = amplitude_to_db(mT_true)
                ys_pred = amplitude_to_db(ys_pred)
                ys_true = amplitude_to_db(ys_true)

            normdiffspec = abs((ys_pred - ys_true) / (mT_true + mT_shift))
            normdiffspec_squared = normdiffspec**2
            loss = torch.mean(normdiffspec_squared)
        else:
            mT_pred = compute_masking_threshold(
                ys_pred, fs, N, nfilts, use_LTQ=use_LTQ, ref_dB=ref_dB
            )
            loss = F.mse_loss(mT_pred, mT_true)
        return loss

    if channels == 1:
        # Mono audio
        mse_loss = compute_channel_loss(
            ys_pred,
            ys_true,
            use_weighting=use_weighting,
            use_LTQ=use_LTQ,
            mT_shift=mT_shift,
            ref_dB=ref_dB,
        )
    else:
        # Stereo audio
        mse_left = compute_channel_loss(
            ys_pred[:, 0, :, :],
            ys_true[:, 0, :, :],
            use_weighting=use_weighting,
            use_LTQ=use_LTQ,
            mT_shift=mT_shift,
            ref_dB=ref_dB,
        )
        mse_right = compute_channel_loss(
            ys_pred[:, 1, :, :],
            ys_true[:, 1, :, :],
            use_weighting=use_weighting,
            use_LTQ=use_LTQ,
            mT_shift=mT_shift,
            ref_dB=ref_dB,
        )
        mse_loss = (mse_left + mse_right) / 2  # Average loss across channels

    return mse_loss


def get_analysis_params(fs, N, nfilts=64):
    maxfreq = fs / 2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfft = 2 * N  # number of fft subbands

    W = mapping2barkmat(fs, nfilts, nfft)
    spreadingfunctionBarkdB = f_SP_dB(maxfreq, nfilts)
    spreadingfuncmatrix = spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts)

    return W, spreadingfuncmatrix, alpha


def compute_masking_threshold(ys, fs, N, nfilts=64, use_LTQ=False, ref_dB=60):
    W, spreadingfuncmatrix, alpha = get_analysis_params(fs, N, nfilts)
    W = W.to(ys.device)
    ys = ys.squeeze(1)
    M = ys.shape[1]  # number of blocks in the signal

    # Compute mXbark for all frames at once
    mXbark = mapping2bark(torch.abs(ys), W, 2 * N)

    # Compute mTbark for all frames at once
    mTbark = maskingThresholdBark(
        mXbark, spreadingfuncmatrix, alpha, fs, nfilts, use_LTQ=use_LTQ, ref_dB=ref_dB
    )

    return mTbark


def amplitude_to_db(x, ref=1.0, amin=1e-10, top_db=200.0):
    x = x.abs()
    x = torch.clamp(x, min=amin)
    x_db = 20.0 * torch.log10(x / ref)
    x_db = torch.clamp(x_db, min=-top_db)
    return x_db


def compute_STFT(
    x, N, return_amplitude=True, hop_length=512, win_length=2048, normalize=True
):
    ys = torch.stft(
        x,
        n_fft=2 * N,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True,
        window=torch.hann_window(2 * N),
    )

    if normalize:
        ys = ys / N * 2

    if return_amplitude:
        ys = torch.abs(ys)

    return ys


def reconstruct_waveform(
    audio_original,
    fft_recon,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    normalize=False,
):
    audio_original = audio_original.squeeze()
    stft = torch.stft(
        audio_original,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        return_complex=True,
        window=torch.hann_window(n_fft),
    )

    N = n_fft // 2
    fft_recon = fft_recon / 2 * N

    phase = torch.angle(stft)

    complex_fft = torch.polar(fft_recon, phase)

    waveform_recon = torch.istft(
        complex_fft,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=torch.hann_window(n_fft),
    )

    waveform_recon = waveform_recon.squeeze()
    if normalize:
        waveform_recon /= torch.max(torch.abs(waveform_recon))

    return waveform_recon


def gaussian_spreading_function(nfilts, sigma):
    x = np.linspace(-nfilts, nfilts, 2 * nfilts)
    spreadingfunctionBark = np.exp(-(x**2) / (2 * sigma**2))
    spreadingfunctionBarkdB = 20 * np.log10(spreadingfunctionBark + np.finfo(float).eps)
    return spreadingfunctionBarkdB


def hyperbolic_spreading_function(nfilts, a, b):
    x = np.linspace(-nfilts, nfilts, 2 * nfilts)
    spreadingfunctionBark = 1 / (1 + np.exp(-a * (x - b)))
    spreadingfunctionBarkdB = 20 * np.log10(spreadingfunctionBark + np.finfo(float).eps)
    return spreadingfunctionBarkdB


def f_SP_dB(maxfreq, nfilts):
    maxbark = hz2bark(maxfreq)
    spreadingfunctionBarkdB = torch.zeros(2 * nfilts)
    spreadingfunctionBarkdB[0:nfilts] = torch.linspace(-maxbark * 27, -8, nfilts) - 23.5
    spreadingfunctionBarkdB[nfilts : 2 * nfilts] = (
        torch.linspace(0, -maxbark * 12.0, nfilts) - 23.5
    )
    return spreadingfunctionBarkdB


def spreadingfunctionmat(spreadingfunctionBarkdB, alpha, nfilts):
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)
    spreadingfuncmatrix = torch.zeros(nfilts, nfilts)
    for k in range(nfilts):
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[
            (nfilts - k) : (2 * nfilts - k)
        ]
    return spreadingfuncmatrix


def maskingThresholdBark(
    mXbark, spreadingfuncmatrix, alpha, fs, nfilts, use_LTQ=False, ref_dB=60
):
    spreadingfuncmatrix = spreadingfuncmatrix.to(mXbark.device)
    mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
    mTbark = mTbark ** (1.0 / alpha)

    if use_LTQ:
        maxfreq = fs / 2.0
        maxbark = hz2bark(maxfreq)
        step_bark = maxbark / (nfilts - 1)
        barks = torch.arange(0, nfilts) * step_bark
        f = bark2hz(barks) + 1e-6
        LTQ = torch.clip(
            (
                3.64 * (f / 1000.0) ** -0.8
                - 6.5 * torch.exp(-0.6 * (f / 1000.0 - 3.3) ** 2.0)
                + 1e-3 * ((f / 1000.0) ** 4.0)
            ),
            -20,
            120,
        ).to(mXbark.device)
        mTbark = torch.max(mTbark, 10.0 ** ((LTQ - ref_dB) / 20))
    return mTbark


def hz2bark(f):
    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f)
    Brk = 6.0 * torch.arcsinh(f / 600.0)
    return Brk


def bark2hz(Brk):
    if not isinstance(Brk, torch.Tensor):
        Brk = torch.tensor(Brk)
    Fhz = 600.0 * torch.sinh(Brk / 6.0)
    return Fhz


def mapping2barkmat(fs, nfilts, nfft):
    maxbark = hz2bark(fs / 2)
    step_bark = maxbark / (nfilts - 1)
    binbark = hz2bark(torch.linspace(0, (nfft / 2), int(nfft / 2) + 1) * fs / nfft)

    filter_indices = torch.round(binbark / step_bark).unsqueeze(0)
    target_indices = torch.arange(nfilts).unsqueeze(1)

    W = (filter_indices == target_indices).float()

    W_padded = torch.zeros(nfilts, nfft)
    W_padded[:, 0 : int(nfft / 2) + 1] = W

    return W_padded


def mapping2bark(mX, W, nfft):
    """
    mX [batch_size, N+1, frame]
    W [M, 2*N] M=64
    """

    nfreqs = int(nfft / 2)

    # Removing the last frequency band and squaring the magnitude
    mX = mX.transpose(-1, -2)
    mXbark_pre_sqrt = torch.matmul(
        torch.clamp(mX[:, :, :nfreqs], min=0.0) ** (2.0),
        # torch.abs(mX[:, :, :nfreqs]) ** (2.0),
        W[:, :nfreqs].T,
    )
    # Now, take the square root
    mXbark = mXbark_pre_sqrt**0.5

    return mXbark


def mappingfrombarkmat(W, nfft):
    nfreqs = int(nfft / 2)
    W_inv = torch.mm(
        torch.diag(1.0 / (torch.sum(W, dim=1) + 1e-6)).sqrt(), W[:, 0 : nfreqs + 1]
    ).T
    return W_inv


def mappingfrombark(mTbark, W_inv, nfft):
    """
    mTbark [batch_size, N, M]
    W_inv [N, M]
    """

    nfreqs = int(nfft / 2)
    mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T)

    return mT  # Keeping shape [batch size, N, N]


def calculate_bits_from_smr(smr):
    return max(1, int(smr / 6))


def get_energy_from_stft(ys):
    return torch.abs(ys) ** 2


def quantize(signal, num_bits):
    levels = 2**num_bits - 1
    quantized = torch.round(signal * levels) / levels
    return quantized


def recon_stft_from_bark_and_quantize(ys, fs=44100, nfilts=64, mT_dB_shift=0):
    N = ys.shape[1] - 1
    nfft = 2 * N

    W, spreadingfuncmatrix, alpha = get_analysis_params(fs, N, nfilts)
    W = W.to(ys.device)
    mXbark = mapping2bark(torch.abs(ys), W, 2 * N)

    example = mXbark[0]
    plt.figure(figsize=(10, 6))
    plt.imshow(
        example.T,
        aspect="auto",
        origin="lower",
        extent=[0, example.shape[0], 0, example.shape[1]],
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Time Frames")
    plt.ylabel("Bark Scale")
    plt.title("Spectrogram in Bark Scale")
    # plt.show()

    mTbark = maskingThresholdBark(
        mXbark, spreadingfuncmatrix, alpha, fs, nfilts, use_LTQ=False
    )

    W_inv = mappingfrombarkmat(W, nfft)
    mT = mappingfrombark(mTbark, W_inv, nfft).transpose(-1, -2)
    mT_dB = amplitude_to_db(mT)
    mT_dB += mT_dB_shift
    ys_dB = amplitude_to_db(ys)
    smr_dB = ys_dB - mT_dB
    max_ys = torch.full_like(ys, 1)
    max_dB = amplitude_to_db(max_ys)
    noise_dB = ys_dB - smr_dB
    smax_to_mask_ratio_dB = max_dB - noise_dB
    assert torch.allclose(noise_dB, mT_dB, atol=1e-5)

    # get the 100th frame
    frame_index = 100
    ys_frame = ys_dB[0, :, frame_index]
    mT_frame = mT_dB[0, :, frame_index]
    smr_frame = smr_dB[0, :, frame_index]
    smax_to_mask_ratio_frame = smax_to_mask_ratio_dB[0, :, frame_index]

    # plot the spectrum and masking threshold for the 100th frame
    plt.figure(figsize=(10, 6))
    freqs = np.linspace(0, 1024, 1025)  # 1025 frequency bins
    plt.plot(freqs, ys_frame, label="Spectrum")
    plt.plot(freqs, mT_frame, label="Masking Threshold", linestyle="--")
    plt.plot(freqs, smr_frame, label="SMR", linestyle="dotted")
    plt.plot(
        freqs, smax_to_mask_ratio_frame, label="Smax to Mask Ratio", linestyle="--"
    )
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Spectrum and Masking Threshold at Frame={frame_index}, mT_dB_shift={mT_dB_shift} dB"
    )
    plt.legend()
    # plt.show()

    quantization_bits = smax_to_mask_ratio_dB // 6
    # quantization_bits = torch.fill_(quantization_bits, 5)

    # plot quantization bits
    plt.figure(figsize=(10, 6))
    plt.imshow(
        quantization_bits.squeeze().cpu().numpy(),
        aspect="auto",
        origin="lower",
        extent=[
            0,
            quantization_bits.squeeze().shape[1],
            0,
            quantization_bits.squeeze().shape[0],
        ],
    )
    plt.colorbar(label="Quantization Bits")
    plt.xlabel("Time Frames")
    plt.ylabel("Frequency Bins")
    plt.title("Quantization Bits")
    # plt.show()

    quantized_stft = torch.zeros_like(ys)
    # print("quantized_stft", quantized_stft.shape)
    # print("quantization_bits", quantization_bits.shape)
    for i in range(ys.shape[1]):  # bin
        for j in range(ys.shape[2]):  # frame
            # print("ys            [0, i, j]", ys[0, i, j])
            quantized_stft[0, i, j] = quantize(ys[0, i, j], quantization_bits[0, i, j])
            # print("quantized_stft[0, i, j]", quantized_stft[0, i, j])

    # raise ValueError("stop")
    # plt.figure(figsize=(20, 8))

    # plt.subplot(1, 2, 1)
    # plt.title("ys Spectrum")
    # plt.imshow(
    #     20 * np.log10(np.abs(ys.squeeze().cpu().numpy() + 1e-8)),
    #     aspect="auto",
    #     origin="lower",
    #     extent=[0, fs / 2, 0, ys.shape[0]],
    # )
    # plt.colorbar(label="Magnitude (dB)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Frame")

    # plt.subplot(1, 2, 2)
    # plt.title(f"quantized_stft quantized Spectrum at mT_dB_shift={mT_dB_shift} dB")
    # plt.imshow(
    #     20 * np.log10(np.abs(quantized_stft.squeeze().cpu().numpy() + 1e-8)),
    #     aspect="auto",
    #     origin="lower",
    #     extent=[0, fs / 2, 0, quantized_stft.shape[0]],
    # )
    # plt.colorbar(label="Magnitude (dB)")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Frame")

    # plt.show()

    return quantized_stft


def plot_results(ys, fs, N, nfilts=64):
    mT = compute_masking_threshold(ys, fs, N, nfilts)

    # Convert STFT magnitude to dB for visualization
    ys = ys.squeeze()
    mT = mT.squeeze()

    ys_dB = 20 * torch.log10(torch.abs(ys) + 1e-6)
    # Convert masking threshold to dB for visualization
    mT_dB = 20 * torch.log10(mT + 1e-6).transpose(0, 1)

    # print("mT", mT_dB)
    # print("mTbarkquant", mTbarkquant)

    # print("mt_dB", mT_dB.shape)
    # print("ys_dB", ys_dB.shape)

    # Frequency and Time vectors for plotting
    f = np.linspace(0, fs / 2, ys.shape[0])
    t = np.linspace(0, 4, ys.shape[1])

    # Plotting
    plt.figure(figsize=(12, 8))

    # Plot Spectrogram
    plt.subplot(3, 1, 1)
    plt.pcolormesh(t, f, ys_dB.numpy(), shading="gouraud", vmin=0, vmax=60)
    plt.colorbar(label="dB")
    plt.title("Spectrogram")
    plt.ylabel("Frequency (Hz)")

    # Plot Spectrum and Masking Threshold of Middle Frame
    middle_frame_idx = len(t) // 2
    plt.subplot(3, 1, 2)
    # print("ys", ys_dB[:, middle_frame_idx].numpy())
    # print("mt", mT_dB[:, middle_frame_idx].numpy())
    plt.plot(
        f, ys_dB[:, middle_frame_idx].numpy(), color="blue", label="Spectrum", alpha=0.7
    )
    plt.plot(
        f,
        mT_dB[:, middle_frame_idx].numpy(),
        color="red",
        label="Masking Threshold",
        alpha=0.7,
    )
    mTbarkquant = mTbarkquant.squeeze()
    W, _, alpha = get_analysis_params(fs, N, nfilts)
    W_inv = mappingfrombarkmat(W, 2 * N)
    mTbarkquant = mappingfrombark(mTbarkquant, W_inv, 2 * N).transpose(0, 1)
    # print("mTbarkquant", mTbarkquant.shape)
    plt.plot(
        f,
        mTbarkquant[:, middle_frame_idx].numpy(),
        color="green",
        label="Masking Threshold (quantized)",
        alpha=0.7,
    )
    plt.legend()
    plt.title(f"Spectrum and Masking Threshold at t = {t[middle_frame_idx]:.2f} s")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude (dB)")

    # Overlay bark scale center frequencies in blue
    W, _, alpha = get_analysis_params(fs, N, nfilts)
    bark_center_freqs = bark2hz(np.linspace(0, hz2bark(fs / 2), W.shape[0]))
    # for freq in bark_center_freqs:
    #     plt.axhline(y=freq, color="blue", linewidth=0.5, alpha=0.7)

    # Plot Spreading Function
    spreadingfunctionBarkdB = f_SP_dB(fs / 2, W.shape[0])
    # print("spreadingfunctionBarkdB", spreadingfunctionBarkdB)
    spreadingfunctionBarkVoltage = 10.0 ** (spreadingfunctionBarkdB / 20.0 * alpha)

    plt.subplot(3, 1, 3)
    plt.plot(spreadingfunctionBarkVoltage.numpy())
    x_length = len(spreadingfunctionBarkVoltage)
    # plt.axvline(x=x_length // 2, color="red", linestyle="--")
    # plt.axvline(x=x_length - 1, color="red", linestyle="--")
    plt.title("Spreading Function")
    plt.xlabel("Bark Scale")
    plt.ylabel("Amplitude (Voltage)")

    plt.tight_layout()
    plt.savefig("spectrogram_with_masking_threshold.png")


def main():
    # Load audio
    fs = 44100
    N = 1024
    nfilts = 64
    mT_dB_shift = 0
    waveform, sample_rate = torchaudio.load("audio_mp3_align.wav")
    audio_mp3_align = waveform[0]

    waveform, sample_rate = torchaudio.load("audio_original.wav")
    audio_original = waveform[0]

    waveform, sample_rate = torchaudio.load("audio_quantized.wav")
    audio_quantized = waveform[0]

    # Compute STFT
    ys_mp3_align = compute_STFT(audio_mp3_align, N=1024).unsqueeze(0).unsqueeze(0)
    ys_original = compute_STFT(audio_original, N=1024).unsqueeze(0).unsqueeze(0)
    ys_quantized = compute_STFT(audio_quantized, N=1024).unsqueeze(0).unsqueeze(0)

    ys_mp3_align = ys_mp3_align[:, :, :, : ys_original.shape[-1]]

    mse_loss_mp3 = F.mse_loss(ys_mp3_align, ys_original)
    print("mse_loss_mp3", mse_loss_mp3.item())
    mse_loss_quant = F.mse_loss(ys_quantized, ys_original)
    print("mse_loss_quant", mse_loss_quant.item())
    print(ys_mp3_align.shape)
    print(
        "mse_loss_mp3/quant ratio, small is better",
        mse_loss_mp3.item() / mse_loss_quant.item(),
    )

    # single file example with weighting
    mp3_ploss = psycho_acoustic_loss(
        ys_mp3_align,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=True,
        use_LTQ=False,
    )
    print("weighted loss: mp3, original", mp3_ploss.item())

    quant_ploss = psycho_acoustic_loss(
        ys_quantized,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=True,
        use_LTQ=False,
    )
    print("weighted loss: quantized, original", quant_ploss.item())
    print(
        "weighted loss: mp3/quant ratio, small is better",
        mp3_ploss.item() / quant_ploss.item(),
    )

    # single file example without weighting
    mp3_ploss = psycho_acoustic_loss(
        ys_mp3_align,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=False,
        use_LTQ=False,
    )
    print("mt diff loss: mp3, original", mp3_ploss.item())

    quant_ploss = psycho_acoustic_loss(
        ys_quantized,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=False,
        use_LTQ=False,
    )
    print("mt diff loss: quantized, original", quant_ploss.item())
    print(
        "mt diff loss: mp3/quant ratio, small is better",
        mp3_ploss.item() / quant_ploss.item(),
    )

    # single file example with weighting and LTQ
    mp3_ploss = psycho_acoustic_loss(
        ys_mp3_align,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=True,
        use_LTQ=True,
    )
    print("weighted loss + LTQ: mp3, original", mp3_ploss.item())

    quant_ploss = psycho_acoustic_loss(
        ys_quantized,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=True,
        use_LTQ=True,
    )
    print("weighted loss + LTQ: quantized, original", quant_ploss.item())
    print(
        "weighted loss + LTQ: mp3/quant ratio, small is better",
        mp3_ploss.item() / quant_ploss.item(),
    )

    # single file example without weighting and with LTQ
    mp3_ploss = psycho_acoustic_loss(
        ys_mp3_align,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=False,
        use_LTQ=True,
    )
    print("mt diff loss + LTQ: mp3, original", mp3_ploss.item())

    quant_ploss = psycho_acoustic_loss(
        ys_quantized,
        ys_original,
        fs=sample_rate,
        N=1024,
        nfilts=64,
        use_weighting=False,
        use_LTQ=True,
    )
    print("mt diff loss + LTQ: quantized, original", quant_ploss.item())
    print(
        "mt diff loss + LTQ: mp3/quant ratio, small is better",
        mp3_ploss.item() / quant_ploss.item(),
    )

    # Plot results
    # plot_results(ys, fs, N, nfilts)

    # fft_recon = recon_fft_from_bark(ys_original.squeeze(0), fs=fs)

    # n_samples = audio_original.shape[0]
    # t = torch.linspace(0, n_samples / fs, n_samples)
    # freq = 440
    # sine_wave = torch.sin(2 * torch.pi * freq * t)
    # sine_wave = (
    #     2 * (sine_wave - sine_wave.min()) / (sine_wave.max() - sine_wave.min()) - 1
    # )
    # ys_original = compute_STFT(sine_wave, N=1024).unsqueeze(0).unsqueeze(0)
    # print("ys_original", ys_original.max())  # 1.4517
    # print("sinewave", sine_wave.min())
    # raise ValueError("stop")

    stft_recon_quant = recon_stft_from_bark_and_quantize(
        ys_original.squeeze(0), fs=fs, nfilts=nfilts, mT_dB_shift=mT_dB_shift
    )
    # waveform_recon = reconstruct_waveform(audio_original, fft_recon)

    waveform_recon_quant = reconstruct_waveform(audio_original, stft_recon_quant)
    # print("waveform_recon", waveform_recon.shape)
    print("waveform_original", audio_original.shape)
    print("waveform_recon_quant", waveform_recon_quant.shape)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(audio_original.numpy())
    plt.title("Original Audio")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.plot(waveform_recon_quant.numpy())
    plt.title(f"Reconstructed Audio at mT_dB_shift={mT_dB_shift} dB")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    # plt.show()

    torchaudio.save(
        "audio_mp3_align_recon_" + str(mT_dB_shift) + ".wav",
        waveform_recon_quant.unsqueeze(0),
        sample_rate,
    )


def test(folder_path):
    # Load audio
    fs = 44100
    N = 1024
    nfilts = 64
    mT_dB_shift = 0

    output_folder = "reconstructed_audio"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(folder_path):
        if file.endswith(".flac") or file.endswith(".wav") or file.endswith(".mp3"):
            filename = file.split(".")[0]
            print(filename)
            waveform, sample_rate = torchaudio.load(os.path.join(folder_path, file))
            audio_original = waveform[0][: fs * 5]  # only take the first 5 seconds
            torchaudio.save(
                os.path.join(output_folder, file.split(".")[0] + "_original.wav"),
                audio_original.unsqueeze(0),
                sample_rate,
            )

            # Compute STFT
            ys_original = compute_STFT(audio_original, N=1024).unsqueeze(0).unsqueeze(0)

            stft_recon_quant = recon_stft_from_bark_and_quantize(
                ys_original.squeeze(0), fs=fs, nfilts=nfilts, mT_dB_shift=mT_dB_shift
            )

            stft_differece = ys_original.squeeze(0) - stft_recon_quant

            plt.figure(figsize=(20, 8))

            plt.subplot(1, 3, 1)
            plt.title("ys Spectrogram")
            plt.imshow(
                20 * np.log10(np.abs(ys_original.squeeze().cpu().numpy() + 1e-8)),
                aspect="auto",
                origin="lower",
                extent=[0, ys_original.shape[3], 0, fs / 2],
            )
            plt.colorbar()
            plt.xlabel("Frame")
            plt.ylabel("Frequency (Hz)")

            plt.subplot(1, 3, 2)
            plt.title(
                f"quantized_stft quantized Spectrum at mT_dB_shift={mT_dB_shift} dB"
            )
            plt.imshow(
                20 * np.log10(np.abs(stft_recon_quant.squeeze().cpu().numpy() + 1e-8)),
                aspect="auto",
                origin="lower",
                extent=[0, ys_original.shape[3], 0, fs / 2],
            )
            plt.colorbar()
            plt.xlabel("Frame")
            plt.ylabel("Frequency (Hz)")

            plt.subplot(1, 3, 3)
            plt.title(f"stft_differece at mT_dB_shift={mT_dB_shift} dB")
            plt.imshow(
                20 * np.log10(np.abs(stft_differece.squeeze().cpu().numpy() + 1e-8)),
                aspect="auto",
                origin="lower",
                extent=[0, ys_original.shape[3], 0, fs / 2],
            )
            plt.colorbar(label="Magnitude (dB)")
            plt.xlabel("Frame")
            plt.ylabel("Frequency (Hz)")
            plt.savefig(os.path.join(output_folder, file.split(".")[0] + "_stft.png"))

            # plot spectrum
            middle_frame = ys_original.shape[3] // 2
            frequencies = np.linspace(0, fs / 2, ys_original.shape[2])

            plt.figure(figsize=(20, 8))

            plt.subplot(1, 3, 1)
            plt.title("ys Spectrum")
            spectrum_ys = 20 * np.log10(
                np.abs(ys_original.squeeze().cpu().numpy()[:, middle_frame] + 1e-8)
            )
            plt.plot(frequencies, spectrum_ys)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            plt.subplot(1, 3, 2)
            plt.title(f"Quantized STFT Spectrum at mT_dB_shift={mT_dB_shift} dB")
            spectrum_quant = 20 * np.log10(
                np.abs(stft_recon_quant.squeeze().cpu().numpy()[:, middle_frame] + 1e-8)
            )
            plt.plot(frequencies, spectrum_quant)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            spectrum_diff = spectrum_ys - spectrum_quant

            plt.subplot(1, 3, 3)
            plt.title(f"Spectrum Difference at mT_dB_shift={mT_dB_shift} dB")
            plt.plot(frequencies, spectrum_diff)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")

            plt.savefig(
                os.path.join(output_folder, file.split(".")[0] + "_spectrum.png")
            )

            waveform_recon_quant = reconstruct_waveform(
                audio_original, stft_recon_quant
            )

            if audio_original.shape[0] != waveform_recon_quant.shape[0]:
                # pad
                waveform_recon_quant = F.pad(
                    waveform_recon_quant,
                    (0, audio_original.shape[0] - waveform_recon_quant.shape[0]),
                )
            audio_difference = audio_original.numpy() - waveform_recon_quant.numpy()

            plt.figure(figsize=(12, 8))
            plt.subplot(3, 1, 1)
            plt.plot(audio_original.numpy())
            plt.title("Original Audio")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            plt.subplot(3, 1, 2)
            plt.plot(waveform_recon_quant.numpy())
            plt.title(f"Reconstructed Audio at mT_dB_shift={mT_dB_shift} dB")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude")

            plt.subplot(3, 1, 3)
            plt.plot(audio_difference)
            plt.title("Difference Audio")
            plt.xlabel("Sample")
            plt.ylabel("Amplitude Difference")

            plt.tight_layout()
            plt.savefig(os.path.join(output_folder, file.split(".")[0] + "_recon.png"))

            torchaudio.save(
                os.path.join(
                    output_folder,
                    file.split(".")[0] + "_recon_" + str(mT_dB_shift) + ".wav",
                ),
                waveform_recon_quant.unsqueeze(0),
                sample_rate,
            )
            plt.close("all")


if __name__ == "__main__":
    main()
    # test("/Users/yyf/Downloads/SQAM_FLAC")
