"""Word-level transcription with a cached faster-whisper model.

Runs inside an interpreter that already has faster-whisper (by default the
Hermes venv). No model download: HF offline mode + local_files_only. The device
actually used is written to the output so a CPU run is never reported as GPU.

usage: transcribe_gpu.py AUDIO OUT_JSON [--model small] [--device cuda]
                         [--cuda-dll-dir DIR] [--allow-cpu-fallback]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

DEFAULT_DLL_DIR = Path(r"C:\Users\sgife\dev\local-agent-runtime\tools")


def add_cuda_dlls(dll_dir):
    if dll_dir and Path(dll_dir).is_dir():
        os.add_dll_directory(str(dll_dir))
        os.environ["PATH"] = f"{dll_dir};{os.environ.get('PATH', '')}"


def run(audio, model_name, device, language):
    from faster_whisper import WhisperModel

    compute = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_name, device=device, compute_type=compute, local_files_only=True)
    segments, info = model.transcribe(
        str(audio), language=language, beam_size=5, word_timestamps=True, vad_filter=False
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "text": w.word.strip()})
    return {"language": info.language, "compute_type": compute, "words": words}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--model", default="small")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--language", default="en")
    ap.add_argument("--cuda-dll-dir", default=os.environ.get("HOOKHAUS_CUDA_DLL_DIR", str(DEFAULT_DLL_DIR)))
    ap.add_argument("--allow-cpu-fallback", action="store_true")
    args = ap.parse_args()

    add_cuda_dlls(args.cuda_dll_dir)
    t0 = time.time()
    device = args.device
    try:
        result = run(args.audio, args.model, device, args.language)
    except RuntimeError as exc:
        if device != "cuda" or not args.allow_cpu_fallback:
            raise
        print(f"CUDA failed ({exc}); CPU fallback explicitly allowed", file=sys.stderr)
        device = "cpu"
        result = run(args.audio, args.model, device, args.language)
    result.update(
        {"model": f"faster-whisper-{args.model} (local cache)", "device": device, "seconds": round(time.time() - t0, 1)}
    )
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"device": device, "words": len(result["words"]), "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
