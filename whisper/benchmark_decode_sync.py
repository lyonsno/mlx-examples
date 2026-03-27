"""Benchmark end-to-end transcription under two decode-loop variants.

This compares full ``mlx_whisper.transcribe(...)`` wall-clock time while
monkey-patching the decode loop between:
1. a patched loop that forces completion with ``mx.eval(next_completed)``; and
2. the current async-only upstream loop.

Use real speech audio via ``--audio``.
"""

import argparse
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time

import mlx.core as mx
import numpy as np

import mlx_whisper
from mlx_whisper.decoding import DecodingTask


def _main_loop_sync(self, audio_features, tokens):
    """Patched loop: force completion once per iteration."""
    n_batch = tokens.shape[0]
    sum_logprobs = mx.zeros(n_batch)

    def _step(inputs, audio_features, tokens, sum_logprobs):
        pre_logits = self.inference.logits(inputs, audio_features)
        logits = pre_logits[:, -1]
        for logit_filter in self.logit_filters:
            logits = logit_filter.apply(logits, tokens)
        tokens, completed, sum_logprobs = self.decoder.update(
            tokens, logits, sum_logprobs
        )
        return tokens, completed, sum_logprobs, pre_logits

    tokens, completed, sum_logprobs, pre_logits = _step(
        tokens, audio_features, tokens, sum_logprobs
    )
    if self.tokenizer.no_speech is not None:
        probs_at_sot = mx.softmax(pre_logits[:, self.sot_index], axis=-1)
        no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
    else:
        no_speech_probs = mx.full(n_batch, mx.nan)
    mx.async_eval(completed, tokens, sum_logprobs, no_speech_probs)

    for _ in range(1, self.sample_len):
        inputs = tokens[:, -1:]
        if tokens.shape[-1] > self.n_ctx:
            break
        next_tokens, next_completed, next_sum_logprobs, _ = _step(
            inputs, audio_features, tokens, sum_logprobs
        )
        mx.eval(next_completed)
        if completed:
            break
        tokens = next_tokens
        completed = next_completed
        sum_logprobs = next_sum_logprobs
        mx.async_eval(next_tokens, next_sum_logprobs)

    return tokens, sum_logprobs, no_speech_probs


def _main_loop_async(self, audio_features, tokens):
    """Current upstream loop: async eval without a sync barrier."""
    n_batch = tokens.shape[0]
    sum_logprobs = mx.zeros(n_batch)

    def _step(inputs, audio_features, tokens, sum_logprobs):
        pre_logits = self.inference.logits(inputs, audio_features)
        logits = pre_logits[:, -1]
        for logit_filter in self.logit_filters:
            logits = logit_filter.apply(logits, tokens)
        tokens, completed, sum_logprobs = self.decoder.update(
            tokens, logits, sum_logprobs
        )
        return tokens, completed, sum_logprobs, pre_logits

    tokens, completed, sum_logprobs, pre_logits = _step(
        tokens, audio_features, tokens, sum_logprobs
    )
    if self.tokenizer.no_speech is not None:
        probs_at_sot = mx.softmax(pre_logits[:, self.sot_index], axis=-1)
        no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
    else:
        no_speech_probs = mx.full(n_batch, mx.nan)
    mx.async_eval(completed, tokens, sum_logprobs, no_speech_probs)

    for _ in range(1, self.sample_len):
        inputs = tokens[:, -1:]
        if tokens.shape[-1] > self.n_ctx:
            break
        next_tokens, next_completed, next_sum_logprobs, _ = _step(
            inputs, audio_features, tokens, sum_logprobs
        )
        mx.async_eval(next_completed, next_tokens, next_sum_logprobs)
        if completed:
            break
        tokens = next_tokens
        completed = next_completed
        sum_logprobs = next_sum_logprobs

    return tokens, sum_logprobs, no_speech_probs


def get_machine_info():
    """Collect stable machine identifiers for the report."""
    hostname = socket.gethostname()
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
        ).strip()
    except Exception:
        chip = "unknown"
    ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    ram_gb = ram_bytes / (1024**3)
    return {
        "hostname": hostname,
        "chip": chip,
        "ram_gb": round(ram_gb),
        "platform": platform.platform(),
    }


def get_audio_source_info(path):
    """Collect metadata about the source audio file."""
    stat = os.stat(path)

    sha256 = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)

    duration_sec = None
    try:
        duration_raw = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            text=True,
        ).strip()
        if duration_raw:
            duration_sec = float(duration_raw)
    except Exception:
        duration_sec = None

    return {
        "path": os.path.abspath(path),
        "bytes": stat.st_size,
        "sha256": sha256.hexdigest(),
        "duration_sec": duration_sec,
    }


def effective_duration(requested_duration_sec, source_duration_sec):
    """Clamp requested duration to the source duration when known."""
    if source_duration_sec is None:
        return requested_duration_sec
    return min(requested_duration_sec, source_duration_sec)


def load_and_trim_audio(path, duration_sec, sr=16000):
    """Load audio with ffmpeg and trim to the requested duration."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        path,
        "-t",
        str(duration_sec),
        "-threads",
        "0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    output = subprocess.run(cmd, capture_output=True, check=True).stdout
    pcm = np.frombuffer(output, np.int16)
    return pcm.astype(np.float32) / 32768.0


def benchmark_variant(name, main_loop_fn, model_repo, audio, n_runs=5):
    """Run one variant n_runs times and return end-to-end transcribe times in ms."""
    original = DecodingTask._main_loop
    DecodingTask._main_loop = main_loop_fn

    times = []
    try:
        for run in range(n_runs):
            mx.eval(mx.zeros(1))

            started = time.monotonic()
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=model_repo,
                language="en",
                verbose=False,
            )
            elapsed_ms = (time.monotonic() - started) * 1000
            text = result.get("text", "").strip()
            times.append(elapsed_ms)
            print(
                f"  {name} run {run + 1}: {elapsed_ms:7.1f}ms  "
                f"({len(text)} chars)"
            )
    finally:
        DecodingTask._main_loop = original

    return times


def run_single(model_repo, requested_duration_sec, n_runs, audio_path, audio_source):
    """Benchmark one model at one duration bucket."""
    used_duration_sec = effective_duration(
        requested_duration_sec,
        audio_source["duration_sec"],
    )

    print("\n" + ("-" * 60))
    print(f"Model: {model_repo}")
    print(f"Requested duration: {requested_duration_sec}s | Runs: {n_runs}")
    if used_duration_sec != requested_duration_sec:
        print(f"Effective duration: {used_duration_sec:.2f}s (source shorter)")
    print("-" * 60)

    # Warm the model before timing.
    dummy = np.zeros(16000, dtype=np.float32)
    mlx_whisper.transcribe(dummy, path_or_hf_repo=model_repo, language="en")

    audio = load_and_trim_audio(audio_path, used_duration_sec)

    print("\nSYNC (patched):")
    sync_times = benchmark_variant("sync", _main_loop_sync, model_repo, audio, n_runs)

    print("\nASYNC (original):")
    async_times = benchmark_variant(
        "async",
        _main_loop_async,
        model_repo,
        audio,
        n_runs,
    )

    if n_runs > 2:
        sync_steady = sync_times[1:]
        async_steady = async_times[1:]
    else:
        sync_steady = sync_times
        async_steady = async_times

    sync_mean = np.mean(sync_steady)
    async_mean = np.mean(async_steady)
    diff_ms = sync_mean - async_mean
    diff_pct = (diff_ms / async_mean) * 100 if async_mean > 0 else 0
    direction = "slower" if diff_ms > 0 else "faster"

    print(f"\n  SYNC:  {sync_mean:7.1f}ms +/- {np.std(sync_steady):5.1f}ms")
    print(f"  ASYNC: {async_mean:7.1f}ms +/- {np.std(async_steady):5.1f}ms")
    print(f"  Sync is {abs(diff_ms):.1f}ms ({abs(diff_pct):.1f}%) {direction}")

    return {
        "model": model_repo,
        "audio_source_path": audio_source["path"],
        "requested_duration_sec": requested_duration_sec,
        "effective_duration_sec": round(used_duration_sec, 3),
        "n_runs": n_runs,
        "sync_times_ms": sync_times,
        "async_times_ms": async_times,
        "sync_steady_mean_ms": round(sync_mean, 1),
        "async_steady_mean_ms": round(async_mean, 1),
        "diff_ms": round(diff_ms, 1),
        "diff_pct": round(diff_pct, 1),
        "direction": direction,
    }


SWEEP_MODELS = [
    "mlx-community/whisper-medium.en-mlx-4bit",
    "mlx-community/whisper-medium.en-mlx-8bit",
    "mlx-community/whisper-medium.en-mlx",
    "mlx-community/whisper-large-v3-turbo-4bit",
    "mlx-community/whisper-large-v3-turbo-8bit",
    "mlx-community/whisper-large-v3-turbo",
]

SWEEP_DURATIONS = [5, 10, 15, 30, 60, 120, 180]


def duration_label(result):
    requested = result["requested_duration_sec"]
    used = result.get("effective_duration_sec", requested)

    def format_duration(value):
        if abs(value - round(value)) < 1e-9:
            return f"{value:.0f}"
        return f"{value:.1f}"

    if abs(requested - used) < 1e-9:
        return f"{format_duration(used)}s"
    return f"{format_duration(requested)}->{format_duration(used)}s"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark transcription wall time for decode sync variants"
    )
    parser.add_argument("--audio", required=True, help="Speech audio file to benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Runs per variant")
    parser.add_argument("--model", type=str, default=None, help="HF model repo")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Requested audio duration in seconds",
    )
    parser.add_argument("--sweep", action="store_true", help="Run all models x durations")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        print(f"Audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(2)

    machine = get_machine_info()
    audio_source = get_audio_source_info(args.audio)
    results = []

    print(f"Machine: {machine['hostname']} - {machine['chip']} - {machine['ram_gb']}GB")
    print(f"Audio source: {audio_source['path']}")
    print(f"Audio bytes: {audio_source['bytes']}")
    if audio_source["duration_sec"] is not None:
        print(f"Audio duration: {audio_source['duration_sec']:.2f}s")
    print(f"Audio sha256: {audio_source['sha256']}")

    if args.sweep:
        for model in SWEEP_MODELS:
            for duration_sec in SWEEP_DURATIONS:
                try:
                    results.append(
                        run_single(
                            model,
                            duration_sec,
                            args.runs,
                            audio_path=args.audio,
                            audio_source=audio_source,
                        )
                    )
                except Exception as exc:
                    print(f"\n  FAILED: {model} @ {duration_sec}s - {exc}")
                    results.append(
                        {
                            "model": model,
                            "audio_source_path": audio_source["path"],
                            "requested_duration_sec": duration_sec,
                            "effective_duration_sec": round(
                                effective_duration(
                                    duration_sec, audio_source["duration_sec"]
                                ),
                                3,
                            ),
                            "error": str(exc),
                        }
                    )
    else:
        model = args.model or "mlx-community/whisper-large-v3-turbo"
        results.append(
            run_single(
                model,
                args.duration,
                args.runs,
                audio_path=args.audio,
                audio_source=audio_source,
            )
        )

    print("\n" + ("=" * 78))
    print(f"SUMMARY - {machine['hostname']} ({machine['chip']}, {machine['ram_gb']}GB)")
    print("=" * 78)
    print(f"{'Model':>45s} {'Dur':>12s} {'Sync':>8s} {'Async':>8s} {'Diff':>10s}")
    print(f"{'-' * 45} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 10}")
    for result in results:
        label = duration_label(result)
        if "error" in result:
            print(f"{result['model']:>45s} {label:>12s} {'FAILED':>8s}")
            continue
        sign = "+" if result["diff_ms"] > 0 else ""
        print(
            f"{result['model']:>45s} {label:>12s} "
            f"{result['sync_steady_mean_ms']:>7.0f}ms "
            f"{result['async_steady_mean_ms']:>7.0f}ms "
            f"{sign}{result['diff_ms']:>6.0f}ms ({sign}{result['diff_pct']:.1f}%)"
        )

    output_path = args.output or f"bench_results_{machine['hostname']}.json"
    report = {
        "machine": machine,
        "audio_source": audio_source,
        "results": results,
    }
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
