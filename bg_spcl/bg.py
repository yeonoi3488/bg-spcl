import numpy as np
from scipy.fft import fft, fftfreq


def compute_at_ratio(segments, args): # compute Alpha-Theta ratio of the given EEG segment

    freqs = fftfreq(args.sampling_rate, 1 / args.sampling_rate)
    alpha_indices = (freqs >= 8) & (freqs < 13)
    theta_indices = (freqs >= 4) & (freqs < 8)

    c_scores = []
    for ind, seg in enumerate(segments): 
        fft_results = fft(seg) # (13, 512)
        psd = np.abs(fft_results) ** 2

        alpha_power = np.mean(psd[:, alpha_indices]) 
        theta_power = np.mean(psd[:, theta_indices])
        c_score = alpha_power / theta_power
        c_scores.append(c_score)

    mean_c_score = np.mean(c_scores)

    return mean_c_score


def compute_bg_scores(args, X): # compute the normalized user state score across trials

    scores_per_trial, indices_to_remove = [], []
    for ind, trial in enumerate(X):
        segments =[]

        for s in range(args.trial_len - 1):
            seg = trial[:, (s+1)*args.sampling_rate: (s+2)*args.sampling_rate]
            segments.append(seg)
            
        mean_c_score = compute_at_ratio(segments, args)
            
        scores_per_trial.append(mean_c_score)

    # min-max normalization
    scores_per_trial = (scores_per_trial - np.min(scores_per_trial)) / (np.max(scores_per_trial) - np.min(scores_per_trial))

    return scores_per_trial

