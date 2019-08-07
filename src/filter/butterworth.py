from scipy.signal import butter, lfilter


def butter_bandpass(low_cut, high_cut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y
