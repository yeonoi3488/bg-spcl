from scipy.signal import butter, filtfilt, resample
from sklearn.preprocessing import StandardScaler


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def downsample_eeg(data, orig_fs=512, target_fs=250):
    num_samples = int(data.shape[-1] * (target_fs / orig_fs))
    return resample(data, num_samples, axis=-1)


def standardize_data(X_train, X_test):
    for j in range(X_train.shape[1]):  
        scaler = StandardScaler()  
        scaler.fit(X_train[:, j, :])  
        X_train[:, j, :] = scaler.transform(X_train[:, j, :])  
        X_test[:, j, :] = scaler.transform(X_test[:, j, :])  
    
    return X_train, X_test