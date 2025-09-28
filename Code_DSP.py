import wfdb
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz, tf2zpk, group_delay, find_peaks
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# STEP 1: Enhanced Data Loading and Visualization
# =============================================================================
def load_ecg_data(record_name='100', samp_to=3600):
    """Load ECG data with comprehensive metadata extraction"""
    print(f"\n{'=' * 50}\nLoading ECG record {record_name}...\n{'=' * 50}")

    record = wfdb.rdrecord(record_name, sampto=samp_to, pn_dir='mitdb')
    annotations = wfdb.rdann(record_name, 'atr', sampto=samp_to, pn_dir='mitdb')

    ecg_signal = record.p_signal[:, 0]
    fs = record.fs
    true_peaks = annotations.sample
    signal_units = record.units[0]

    print(f"Record: {record_name}")
    print(f"Sampling frequency: {fs} Hz")
    print(f"Duration: {len(ecg_signal) / fs:.2f} sec")
    print(f"True R-peaks count: {len(true_peaks)}")
    print(f"Signal units: {signal_units}")

    # Plot original signal
    plt.figure(figsize=(15, 5))
    plt.plot(ecg_signal, label='ECG Signal')
    plt.scatter(true_peaks, ecg_signal[true_peaks], color='red',
                s=30, label='True R-peaks', alpha=0.7)
    plt.title(f'Original ECG Signal (Record {record_name})', fontsize=14)
    plt.xlabel('Samples')
    plt.ylabel(f'Amplitude ({signal_units})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return ecg_signal, fs, true_peaks


# =============================================================================
# STEP 2: Enhanced Pan-Tompkins Implementation
# =============================================================================
class EnhancedPanTompkins:
    def __init__(self, fs=360):
        self.fs = fs
        self.bandpass_cutoff = [5, 15]  # Hz
        self.integration_window = 0.15  # seconds
        self.refractory_period = 0.2  # seconds

        # Initialize filters
        self.b_bp, self.a_bp = self._create_bandpass_filter()
        self.derivative_kernel = np.array([-1, -2, 0, 2, 1]) * (fs / 8)

    def _create_bandpass_filter(self):
        """Create Butterworth bandpass filter"""
        nyq = 0.5 * self.fs
        low = self.bandpass_cutoff[0] / nyq
        high = self.bandpass_cutoff[1] / nyq
        b, a = butter(2, [low, high], btype='band')
        return b, a

    def process(self, ecg_signal):
        """Complete Pan-Tompkins signal processing pipeline"""
        # 1. Bandpass filtering
        filtered = filtfilt(self.b_bp, self.a_bp, ecg_signal)

        # 2. Derivative
        derivative = np.convolve(filtered, self.derivative_kernel, mode='same')

        # 3. Squaring
        squared = derivative ** 2

        # 4. Moving window integration
        window_size = int(self.integration_window * self.fs)
        integrated = np.convolve(squared, np.ones(window_size) / window_size, mode='same')

        return {
            'filtered': filtered,
            'derivative': derivative,
            'squared': squared,
            'integrated': integrated
        }


# =============================================================================
# STEP 3: Advanced Thresholding with LMS and Dynamic Adaptation
# =============================================================================
class AdaptiveThresholdDetector:
    def __init__(self, fs=360):
        self.fs = fs
        self.min_peak_distance = int(0.2 * fs)

        # Threshold parameters
        self.spki = 0.6  # Running estimate of signal peaks
        self.npki = 0.2  # Running estimate of noise peaks
        self.threshold_i1 = self.npki + 0.25 * (self.spki - self.npki)
        self.threshold_i2 = 0.5 * self.threshold_i1

        # LMS adaptive parameters
        self.mu = 0.01  # Learning rate
        self.weights = np.array([0.5, 0.3, 0.2])  # Weighting for peak history

        # Peak history
        self.peak_history = []
        self.max_history = 5

    def update_thresholds(self, peak_value):
        """Update thresholds using combined Pan-Tompkins and LMS approach"""
        # Pan-Tompkins logic
        if peak_value > self.threshold_i1:
            self.spki = 0.125 * peak_value + 0.875 * self.spki
        elif peak_value > self.threshold_i2:
            self.spki = 0.25 * peak_value + 0.75 * self.spki
        else:
            self.npki = 0.125 * peak_value + 0.875 * self.npki

        # Update thresholds
        self.threshold_i1 = self.npki + 0.25 * (self.spki - self.npki)
        self.threshold_i2 = 0.5 * self.threshold_i1

        # LMS adaptation
        if len(self.peak_history) >= 3:
            features = np.array(self.peak_history[-3:])
            prediction = np.dot(self.weights, features)
            error = (1 if peak_value > self.threshold_i1 else 0) - prediction
            self.weights += self.mu * error * features

        # Update peak history
        self.peak_history.append(peak_value)
        if len(self.peak_history) > self.max_history:
            self.peak_history.pop(0)

        return self.threshold_i1, self.threshold_i2

    def detect_peaks(self, processed_signal):
        """Detect QRS peaks with adaptive thresholding"""
        # Find all potential peaks
        peaks, properties = find_peaks(processed_signal,
                                       distance=self.min_peak_distance,
                                       height=0.1)

        if len(peaks) == 0:
            return np.array([]), [], []

        peak_values = processed_signal[peaks]
        # Normalize peak values to be between 0 and 1 for consistent threshold updates
        # This is a key change for adaptive thresholds to work reliably across different signal amplitudes.
        # If the original signal values are very large/small, the thresholds might not adapt well.
        # However, if your signals are already normalized by the Pan-Tompkins steps (e.g., squaring amplifies),
        # then direct use might be fine. For robustness, normalizing here.
        normalized_peaks = peak_values / np.max(processed_signal) if np.max(processed_signal) != 0 else peak_values

        detected_peaks = []
        thresholds_i1_history = []
        thresholds_i2_history = []

        for i, peak_idx in enumerate(peaks):
            peak_val = normalized_peaks[i] # Use normalized peak value for threshold update
            t1, t2 = self.update_thresholds(peak_val)
            thresholds_i1_history.append(t1)
            thresholds_i2_history.append(t2)

            # Compare original peak value with threshold scaled back to original signal range
            # The thresholds (t1, t2) are based on normalized values, so scale them back to compare with `processed_signal`
            if peak_val * np.max(processed_signal) > t1 * np.max(processed_signal):
                detected_peaks.append(peak_idx)

        return np.array(detected_peaks), thresholds_i1_history, thresholds_i2_history

    def detect_peaks_static(self, processed_signal, static_threshold_factor=0.6):
        """Detect QRS peaks using a static threshold."""
        # Find all potential peaks
        peaks, properties = find_peaks(processed_signal,
                                       distance=self.min_peak_distance,
                                       height=0.1) # Initial height to get candidates

        if len(peaks) == 0:
            return np.array([]), 0

        max_signal_val = np.max(processed_signal)
        static_threshold = static_threshold_factor * max_signal_val

        detected_peaks_static = peaks[processed_signal[peaks] > static_threshold]

        return detected_peaks_static, static_threshold


# =============================================================================
# STEP 4: Comprehensive DSP Analysis Tools
# =============================================================================
class DSPAnalyzer:
    @staticmethod
    def analyze_filter(b, a, fs, title):
        """Comprehensive filter analysis with 4 plots"""
        plt.figure(figsize=(15, 10))

        # Frequency response
        w, h = freqz(b, a, worN=8000)
        freqs = w * fs / (2 * np.pi)

        # Magnitude response
        plt.subplot(2, 2, 1)
        plt.plot(freqs, 20 * np.log10(np.abs(h)))
        plt.title(f'{title} - Magnitude Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude (dB)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, fs / 2)

        # Phase response
        plt.subplot(2, 2, 2)
        plt.plot(freqs, np.unwrap(np.angle(h)))
        plt.title(f'{title} - Phase Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Phase (radians)')
        plt.grid(True, alpha=0.3)
        plt.xlim(0, fs / 2)

        # Group delay
        plt.subplot(2, 2, 3)
        try:
            w_gd, gd = group_delay((b, a), w=8000)
            plt.plot(w_gd * fs / (2 * np.pi), gd)
            plt.title(f'{title} - Group Delay')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Samples')
            plt.grid(True, alpha=0.3)
            plt.xlim(0, fs / 2)
        except:
            plt.text(0.5, 0.5, 'Group delay unavailable',
                     ha='center', va='center')

        # Pole-zero plot
        plt.subplot(2, 2, 4)
        z, p, k = tf2zpk(b, a)

        plt.scatter(np.real(z), np.imag(z), marker='o',
                    facecolors='none', edgecolors='b', label='Zeros')
        plt.scatter(np.real(p), np.imag(p), marker='x',
                    color='r', label='Poles')
        unit_circle = plt.Circle((0, 0), 1, color='k',
                                 fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(unit_circle)
        plt.title(f'{title} - Pole-Zero Plot')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')

        plt.tight_layout()
        plt.show()

        # Calculate and print filter characteristics
        print(f"\n{title} Characteristics:")
        if len(p) > 0:
            print(f"- Stability: {'Stable' if all(np.abs(p) < 1) else 'Unstable'}")

        if len(b) > 1 and len(a) > 1:
            dc_gain = np.sum(b) / np.sum(a)
            print(f"- DC Gain: {dc_gain:.4f}")

            # Estimate cutoff frequencies
            mag = 20 * np.log10(np.abs(h))
            if np.any(mag < -3):
                cutoff_idx = np.where(mag < -3)[0][0]
                cutoff_freq = freqs[cutoff_idx]
                print(f"- Approx. -3dB Frequency: {cutoff_freq:.2f} Hz")


# =============================================================================
# STEP 5: Performance Evaluation Metrics
# =============================================================================
class ECGEvaluator:
    @staticmethod
    def evaluate(detected_peaks, true_peaks, fs, tolerance_ms=50):
        """Comprehensive evaluation with multiple metrics"""
        tolerance_samples = int(tolerance_ms * fs / 1000)

        TP = 0  # True positives
        FP = 0  # False positives
        FN = 0  # False negatives

        matched_true = []
        matched_detected = []

        # Find matches within tolerance
        for t in true_peaks:
            distances = np.abs(detected_peaks - t)
            within_tolerance = distances <= tolerance_samples

            if np.any(within_tolerance):
                TP += 1
                closest_idx = np.argmin(distances)
                matched_true.append(t)
                # Ensure we don't double count a detected peak if it matches multiple true peaks
                # For simplicity, we just add the first closest one, but a more robust matching
                # would involve marking detected peaks as used.
                if detected_peaks[closest_idx] not in matched_detected:
                    matched_detected.append(detected_peaks[closest_idx])
            else:
                FN += 1 # True peak not detected

        # Count false positives (detected peaks that didn't match any true peak)
        fp_candidates = [d for d in detected_peaks if d not in matched_detected]
        FP = len(fp_candidates)


        # Calculate metrics
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        f1_score = 2 * (sensitivity * precision) / (sensitivity + precision) if (sensitivity + precision) > 0 else 0

        # Calculate timing errors for matched peaks
        timing_errors = []
        # Re-match specifically for timing errors to ensure matched_true and matched_detected are correctly paired
        temp_matched_true = []
        temp_matched_detected = []
        for t in true_peaks:
            distances = np.abs(detected_peaks - t)
            within_tolerance = distances <= tolerance_samples
            if np.any(within_tolerance):
                closest_idx = np.argmin(distances)
                temp_matched_true.append(t)
                temp_matched_detected.append(detected_peaks[closest_idx])

        for t, d in zip(temp_matched_true, temp_matched_detected):
            timing_errors.append((d - t) / fs * 1000)  # in ms

        mean_error = np.mean(timing_errors) if timing_errors else 0
        std_error = np.std(timing_errors) if timing_errors else 0

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'Sensitivity': sensitivity,
            'Precision': precision,
            'F1_Score': f1_score,
            'Mean_Error_ms': mean_error,
            'Std_Error_ms': std_error,
            'Matched_True': temp_matched_true, # Use temp_matched_true for accurate list
            'Matched_Detected': temp_matched_detected # Use temp_matched_detected for accurate list
        }

    @staticmethod
    def plot_results(ecg_signal, true_peaks, detected_peaks, evaluation, title):
        """Visualize detection results with performance metrics"""
        plt.figure(figsize=(15, 6))

        # Plot ECG signal
        plt.plot(ecg_signal, label='ECG Signal', alpha=0.7)

        # Plot true peaks
        plt.scatter(true_peaks, ecg_signal[true_peaks],
                    color='green', s=50, label='True R-peaks', alpha=0.7)

        # Plot detected peaks
        plt.scatter(detected_peaks, ecg_signal[detected_peaks],
                    color='red', marker='x', s=100, label='Detected R-peaks')

        # Plot false negatives
        fn_peaks = [t for t in true_peaks if t not in evaluation['Matched_True']]
        if fn_peaks:
            plt.scatter(fn_peaks, ecg_signal[fn_peaks],
                        color='orange', marker='o', s=150,
                        facecolors='none', linewidths=2, label='False Negatives')

        # Plot false positives
        # Re-calculate FP peaks based on the logic in evaluation
        fp_peaks_display = []
        for d_idx in detected_peaks:
            is_matched = False
            for t_idx in true_peaks:
                if np.abs(d_idx - t_idx) <= int(50 * 360 / 1000): # Use the same tolerance as evaluation
                    is_matched = True
                    break
            if not is_matched:
                fp_peaks_display.append(d_idx)

        if fp_peaks_display:
            plt.scatter(fp_peaks_display, ecg_signal[fp_peaks_display],
                        color='purple', marker='s', s=80,
                        facecolors='none', linewidths=2, label='False Positives')

        plt.title(f'{title}\nSensitivity: {evaluation["Sensitivity"]:.2%} | ' +
                  f'Precision: {evaluation["Precision"]:.2%} | ' +
                  f'F1 Score: {evaluation["F1_Score"]:.3f}', fontsize=12)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


# =============================================================================
# STEP 6: Main Processing Pipeline
# =============================================================================
def main():
    # 1. Load data
    ecg_signal, fs, true_peaks = load_ecg_data('100', samp_to=3600)

    # 2. Initialize Pan-Tompkins processor
    pt_processor = EnhancedPanTompkins(fs)

    # 3. Process signal through all stages
    processed = pt_processor.process(ecg_signal)

    # 4. Detect peaks with adaptive thresholding
    detector = AdaptiveThresholdDetector(fs)
    detected_peaks_adaptive, thresholds_i1_adaptive, thresholds_i2_adaptive = \
        detector.detect_peaks(processed['integrated'])

    # NEW: Detect peaks with static thresholding for comparison
    # Choose a static threshold factor (e.g., 0.5, 0.6, 0.7)
    # This factor is applied to the maximum value of the integrated signal.
    static_threshold_factor = 0.6
    detected_peaks_static, static_threshold_value = \
        detector.detect_peaks_static(processed['integrated'], static_threshold_factor=static_threshold_factor)


    # 5. Evaluate performance for ADAPTIVE detection
    evaluator = ECGEvaluator()
    evaluation_adaptive = evaluator.evaluate(detected_peaks_adaptive, true_peaks, fs)

    # NEW: Evaluate performance for STATIC detection
    evaluation_static = evaluator.evaluate(detected_peaks_static, true_peaks, fs)


    # 6. Visualize results for ADAPTIVE detection
    evaluator.plot_results(ecg_signal, true_peaks, detected_peaks_adaptive, evaluation_adaptive,
                           "Enhanced Pan-Tompkins QRS Detection (Adaptive Thresholding)")

    # NEW: Visualize results for STATIC detection
    evaluator.plot_results(ecg_signal, true_peaks, detected_peaks_static, evaluation_static,
                           f"Pan-Tompkins QRS Detection (Static Threshold: {static_threshold_factor*100:.0f}% of Max)")


    # 7. DSP Analysis
    dsp = DSPAnalyzer()
    print("\nPerforming DSP analysis on filters...")

    # Analyze bandpass filter
    dsp.analyze_filter(pt_processor.b_bp, pt_processor.a_bp, fs,
                       "Bandpass Filter (5-15 Hz)")

    # Analyze derivative filter
    dsp.analyze_filter(pt_processor.derivative_kernel, [1], fs,
                       "Derivative Filter")

    # Analyze moving average filter
    window_size = int(pt_processor.integration_window * fs)
    ma_kernel = np.ones(window_size) / window_size
    dsp.analyze_filter(ma_kernel, [1], fs,
                       f"Moving Average Filter (N={window_size})")

    # Print performance summary for Adaptive Detection
    print("\n" + "=" * 60)
    print("Performance Summary: Adaptive Detection".center(60))
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>15} {'Unit':>20}")
    print(f"{'Sensitivity':<20} {evaluation_adaptive['Sensitivity']:>15.2%} {'':>20}")
    print(f"{'Precision':<20} {evaluation_adaptive['Precision']:>15.2%} {'':>20}")
    print(f"{'F1 Score':<20} {evaluation_adaptive['F1_Score']:>15.3f} {'':>20}")
    print(f"{'True Positives':<20} {evaluation_adaptive['TP']:>15} {'':>20}")
    print(f"{'False Positives':<20} {evaluation_adaptive['FP']:>15} {'':>20}")
    print(f"{'False Negatives':<20} {evaluation_adaptive['FN']:>15} {'':>20}")
    print(f"{'Mean Timing Error':<20} {evaluation_adaptive['Mean_Error_ms']:>15.2f} {'ms':>20}")
    print(f"{'Timing Error Std':<20} {evaluation_adaptive['Std_Error_ms']:>15.2f} {'ms':>20}")
    print("=" * 60 + "\n")

    # Print performance summary for Static Detection
    print("\n" + "=" * 60)
    print("Performance Summary: Static Detection".center(60))
    print("=" * 60)
    print(f"{'Metric':<20} {'Value':>15} {'Unit':>20}")
    print(f"{'Sensitivity':<20} {evaluation_static['Sensitivity']:>15.2%} {'':>20}")
    print(f"{'Precision':<20} {evaluation_static['Precision']:>15.2%} {'':>20}")
    print(f"{'F1 Score':<20} {evaluation_static['F1_Score']:>15.3f} {'':>20}")
    print(f"{'True Positives':<20} {evaluation_static['TP']:>15} {'':>20}")
    print(f"{'False Positives':<20} {evaluation_static['FP']:>15} {'':>20}")
    print(f"{'False Negatives':<20} {evaluation_static['FN']:>15} {'':>20}")
    print(f"{'Mean Timing Error':<20} {evaluation_static['Mean_Error_ms']:>15.2f} {'ms':>20}")
    print(f"{'Timing Error Std':<20} {evaluation_static['Std_Error_ms']:>15.2f} {'ms':>20}")
    print("=" * 60 + "\n")


    # Plot all processing stages including both adaptive and static detection on integrated signal
    plt.figure(figsize=(15, 15)) # Increased figure height

    # Store adaptive thresholds corresponding to the full integrated signal
    # This requires running the adaptive threshold logic across the entire signal
    # to get the dynamic threshold values, which is more involved.
    # For now, we will plot the adaptive thresholds only at the detected peak locations
    # and a horizontal line for the static threshold.

    stages = [
        ("Original ECG", ecg_signal, None, None, None, None),
        ("Bandpass Filtered", processed['filtered'], None, None, None, None),
        ("Derivative", processed['derivative'], None, None, None, None),
        ("Squared Signal", processed['squared'], None, None, None, None),
        ("Integrated Signal with Detections",
         processed['integrated'],
         detected_peaks_adaptive, None, # Pass None for adaptive threshold line as it's dynamic
         detected_peaks_static, static_threshold_value)
    ]

    for i, (title, data, adaptive_peaks, adaptive_threshold_line, static_peaks, static_threshold_val) in enumerate(stages, 1):
        plt.subplot(len(stages), 1, i)
        plt.plot(data, color='gray', alpha=0.7) # Plot signal in gray for background

        if title == "Integrated Signal with Detections":
            # Plot adaptive peaks
            if adaptive_peaks is not None and len(adaptive_peaks) > 0:
                plt.scatter(adaptive_peaks, data[adaptive_peaks],
                            color='red', marker='x', s=100, label='Adaptive Detected Peaks')

            # Plot static peaks
            if static_peaks is not None and len(static_peaks) > 0:
                plt.scatter(static_peaks, data[static_peaks],
                            color='blue', marker='o', s=50, facecolors='none', edgecolors='blue',
                            linewidths=1.5, label='Static Detected Peaks')

            # Plot static threshold line
            if static_threshold_val is not None:
                plt.axhline(y=static_threshold_val, color='green', linestyle='--', alpha=0.8,
                            label=f'Static Threshold ({static_threshold_factor*100:.0f}% of Max)')

            # Note: Plotting adaptive_thresholds_i1 as a continuous line is complex
            # because they are updated per peak, not per sample.
            # If you wish to see the dynamic threshold, you would need to store
            # the threshold values at every sample during the detection loop.
            # For this plot, we just show the peaks detected by it.

            plt.legend(loc='upper right', fontsize=8) # Add legend for detected peaks and static threshold
        else:
            # For other stages, just plot the signal.
            plt.plot(data, color='blue') # Or a different color for non-integrated signals

        plt.title(title, fontsize=10)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # =============================================================================
    # STEP 8: Final Performance Comparison Plot (Your Added Code)
    # =============================================================================
    print("Generating final performance comparison chart...")

    # Labels and data
    metrics = ['Sensitivity', 'Precision', 'F1 Score']
    lms_scores = [evaluation_adaptive['Sensitivity'], evaluation_adaptive['Precision'], evaluation_adaptive['F1_Score']]
    static_scores = [evaluation_static['Sensitivity'], evaluation_static['Precision'], evaluation_static['F1_Score']]

    x = np.arange(len(metrics))  # the label locations
    width = 0.35  # width of the bars

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, lms_scores, width, label='LMS Threshold', color='skyblue')
    bars2 = ax.bar(x + width / 2, static_scores, width, label='Static Threshold (60%)', color='orange')

    # Add labels, title, legend
    ax.set_ylabel('Score')
    ax.set_title('Performance Comparison: LMS vs Static Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Add value labels on top of bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()


