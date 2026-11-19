"""Conservative background suppression before the native audio encoder."""

import time

import numpy as np


class PlaybackEchoGuard:
    """Identify playback-correlated frames for VAD, without editing model input.

    Match short waveform windows because browser AEC/AGC and clock drift can
    invalidate one delay/gain estimate over an entire second. Unmatched frames
    remain eligible for the ordinary speech-onset detector, including speech
    over an echo. Each session owns only its recent, actually emitted audio.
    """

    def __init__(self, cfg):
        self.reference = np.empty(0, dtype=np.float64)
        self.history_samples = int(cfg["echo_reference_seconds"] * 16000)
        self.window_samples = int(cfg["echo_match_window_ms"] * 16)
        self.threshold = cfg["echo_min_correlation"]
        self.energy_floor = cfg["vad_min_rms"]
        self.playback_deadline = 0.0

    def add_output(self, waveform, now):
        from scipy.signal import resample_poly

        reference = resample_poly(np.asarray(waveform, dtype=np.float64), 2, 3)
        self.reference = np.concatenate([self.reference, reference])[
            -self.history_samples :
        ]
        self.playback_deadline = (
            max(now + 0.08, self.playback_deadline) + len(waveform) / 24000
        )

    def reset(self):
        self.reference = np.empty(0, dtype=np.float64)
        self.playback_deadline = 0.0

    def classify(self, waveform, queued_seconds, now):
        from scipy.signal import correlate

        started = time.monotonic()
        frames = len(waveform) // 320
        matched = np.zeros(frames, dtype=bool)
        scores = np.zeros(frames, dtype=np.float64)
        # The microphone packet covers the preceding second. Allow transport
        # uncertainty, but never compare against distant conversation audio.
        end = len(self.reference) - int(queued_seconds * 16000)
        reference = self.reference[max(0, end - 24000) : max(0, end + 4000)]
        if len(reference) >= self.window_samples and (
            queued_seconds > 0 or now <= self.playback_deadline + 1.25
        ):
            ones = np.ones(self.window_samples)
            sums = np.convolve(reference, ones, mode="valid")
            energy = np.maximum(
                0,
                np.convolve(reference**2, ones, mode="valid")
                - sums**2 / self.window_samples,
            )
            audible = energy / self.window_samples >= self.energy_floor**2
            for index in range(frames):
                frame = waveform[index * 320 : (index + 1) * 320]
                offset = min(
                    max(0, index * 320 + 160 - self.window_samples // 2),
                    len(waveform) - self.window_samples,
                )
                window = np.asarray(
                    waveform[offset : offset + self.window_samples], dtype=np.float64
                )
                window = window - window.mean()
                power = float(np.dot(window, window))
                if float(np.mean(frame**2)) >= self.energy_floor**2 and power > 0:
                    correlation = correlate(
                        reference, window, mode="valid", method="fft"
                    )
                    normalized = np.where(
                        audible,
                        np.abs(correlation)
                        / np.sqrt(np.maximum(energy * power, 1e-12)),
                        0,
                    )
                    scores[index] = min(1.0, float(normalized.max()))
                    matched[index] = scores[index] >= self.threshold
        return matched, dict(
            matched_frames=int(matched.sum()),
            frame_ms=20,
            max_correlation=float(scores.max(initial=0)),
            threshold=self.threshold,
            reference_samples=len(reference),
            cpu_seconds=round(time.monotonic() - started, 6),
        )


class MicrophoneNoiseFilter:
    """Canonicalize a complete non-speech input unit.

    Speech is checked at original and increased detector gain, independently
    of input energy. Browser AGC can lift room noise above an energy ceiling.
    An uncertain speech detection preserves the entire unit. This filter has
    no access to actions, turn state, output playback, or the language model.
    It neither removes samples nor declares that a user turn has ended.
    """

    def __init__(self, cfg):
        import webrtcvad

        self.vad = webrtcvad.Vad(cfg["input_noise_vad_mode"])
        self.ceiling = cfg["input_noise_floor_rms"]
        self.speech_model = None
        self.speech_threshold = cfg.get("input_noise_speech_probability", 0.3)
        self.min_speech_ms = cfg.get("input_noise_min_speech_ms", 96)
        self.strong_speech_threshold = cfg.get(
            "input_noise_strong_speech_probability", 0.85
        )
        self.detector_gain = cfg.get("input_noise_detector_gain", 10.0)
        if cfg.get("input_noise_speech_model"):
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            self.speech_model = onnxruntime.InferenceSession(
                cfg["input_noise_speech_model"],
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )

    def process(self, waveform):
        filtered = waveform.copy()
        pcm = (np.clip(waveform, -1, 1) * 32767).astype("<i2")
        speech_frames, protected = 0, 0
        for offset in range(0, len(waveform), 320):
            frame = waveform[offset : offset + 320]
            if len(frame) != 320:
                break
            else:
                quiet = float(np.sqrt(np.mean(frame**2))) < self.ceiling
            speech = self.vad.is_speech(pcm[offset : offset + 320].tobytes(), 16000)
            speech_frames += int(speech)
            protected += int(quiet and speech)
        quiet_unit = float(np.sqrt(np.mean(waveform**2))) < self.ceiling
        evidence, original_evidence = None, None
        protected_speech = speech_frames > 0
        if self.speech_model is not None:
            # Classifier-only gain protects low-volume speech. The waveform
            # passed to the native model is never amplified or partly edited.
            # Reset recurrent state per complete unit so preceding speech does
            # not keep a stationary quiet background classified as speech.
            original_evidence = self._speech_evidence(waveform, 1.0)
            evidence = self._speech_evidence(waveform, self.detector_gain)
            # A low-confidence 32/64 ms spike is common in AGC room noise.
            # Brief speech with strong evidence remains protected, as does
            # sustained weaker speech. No individual frames are removed.
            protected_speech = any(
                value["max_probability"] >= self.strong_speech_threshold
                or value["longest_speech_ms"] >= self.min_speech_ms
                for value in (original_evidence, evidence)
            )
            nonspeech_unit = not protected_speech
        else:
            nonspeech_unit = quiet_unit and not protected_speech
        if nonspeech_unit:
            filtered.fill(0)
            suppressed = len(waveform) // 320
        else:
            # Preserve light consonants, pauses, and the exact native input
            # conditioning whenever any part of the unit may contain speech.
            suppressed = 0
        return filtered, dict(
            mode="nonspeech_unit_canonical_silence",
            frame_ms=20,
            suppressed_frames=suppressed,
            protected_quiet_speech_frames=protected,
            speech_frames=speech_frames,
            preserved_entire_unit=suppressed == 0,
            total_frames=len(waveform) // 320,
            ceiling_rms=self.ceiling,
            speech_recheck="silero_cpu" if self.speech_model is not None else "webrtc",
            speech_probability=evidence["max_probability"] if evidence else None,
            original_speech_probability=original_evidence["max_probability"]
            if original_evidence
            else None,
            speech_evidence=evidence,
            original_speech_evidence=original_evidence,
            min_speech_ms=self.min_speech_ms,
            strong_speech_probability=self.strong_speech_threshold,
            detector_gain=self.detector_gain,
        )

    def _speech_evidence(self, waveform, gain):
        memory = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros(64, dtype=np.float32)
        probability = 0.0
        consecutive_ms, longest_ms = 0.0, 0.0
        for offset in range(0, len(waveform), 512):
            frame = waveform[offset : offset + 512] * gain
            duration_ms = len(frame) / 16
            frame = np.pad(frame, (0, 512 - len(frame)))
            inputs = np.concatenate([context, frame])[None, :]
            score, memory = self.speech_model.run(
                None,
                dict(input=inputs, state=memory, sr=np.array(16000, dtype=np.int64)),
            )
            probability = max(probability, float(score.item()))
            consecutive_ms = (
                consecutive_ms + duration_ms
                if float(score.item()) >= self.speech_threshold
                else 0.0
            )
            longest_ms = max(longest_ms, consecutive_ms)
            context = frame[-64:]
        return dict(max_probability=probability, longest_speech_ms=longest_ms)
