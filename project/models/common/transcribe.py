"""Whisper-based ASR transcription with JSON caching."""

import os
import json


# Cache file path (project root)
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "transcriptions.json")


def transcribe_dataset(file_paths, cache_path=None):
    """Transcribe all audio files using Whisper, with JSON caching.

    Args:
        file_paths: list of .wav file paths
        cache_path: path to JSON cache file

    Returns:
        dict mapping file_path -> transcription string
    """
    if cache_path is None:
        cache_path = CACHE_PATH

    cache_path = os.path.abspath(cache_path)

    # Load existing cache
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            cache = json.load(f)

    # Find files that need transcription
    missing = [fp for fp in file_paths if fp not in cache]

    if missing:
        print(f"Transcribing {len(missing)} audio files with Whisper (cached: {len(cache)})...")
        import whisper
        model = whisper.load_model("base")

        for i, fp in enumerate(missing):
            result = model.transcribe(fp, language="en")
            cache[fp] = result["text"].strip()
            if (i + 1) % 100 == 0:
                print(f"  Transcribed {i + 1}/{len(missing)} files...")
                # Save intermediate progress
                with open(cache_path, "w") as f:
                    json.dump(cache, f, indent=2)

        # Save final cache
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"Transcription complete. Cache saved to {cache_path}")
    else:
        print(f"All {len(file_paths)} transcriptions loaded from cache.")

    return cache


def get_transcripts_for_files(file_paths, cache_path=None):
    """Get transcriptions for a list of files, transcribing if needed.

    Args:
        file_paths: list of .wav file paths

    Returns:
        list of transcription strings (same order as file_paths)
    """
    cache = transcribe_dataset(file_paths, cache_path)
    return [cache.get(fp, "") for fp in file_paths]
