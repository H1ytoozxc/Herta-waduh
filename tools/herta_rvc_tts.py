import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch


DEFAULT_APPLIO_ROOT = Path(r'Z:\APPLIO')
DEFAULT_RVC_MODEL_PATH = Path(r'Z:\ГЕРТАААА\model.pth')
FALLBACK_RVC_MODEL_PATH = Path(r'Z:\APPLIO\logs\model\model.pth')
DEFAULT_PITCH = 5
DEFAULT_HERTA_MODEL_DIR = Path('Z:/') / '\u0413\u0415\u0420\u0422\u0410\u0410\u0410\u0410'
DEFAULT_RVC_MODEL_PATH = DEFAULT_HERTA_MODEL_DIR / 'model.pth'
DEFAULT_PITCH = 0
DEFAULT_F0_METHOD = 'rmvpe'
DEFAULT_PITCH_SWEEP = (-2, 0, 3, 5, 7)
DEFAULT_SILERO_REPO = Path.home() / '.cache' / 'torch' / 'hub' / 'snakers4_silero-models_master'


def parse_device(raw_device: str | None) -> int | str | None:
    if raw_device is None:
        return None
    stripped = raw_device.strip()
    if not stripped:
        return None
    return int(stripped) if stripped.isdigit() else stripped


def resolve_rvc_model_path(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path
    if requested_path == DEFAULT_RVC_MODEL_PATH and FALLBACK_RVC_MODEL_PATH.exists():
        print(f"RVC model was not found at {requested_path}; using {FALLBACK_RVC_MODEL_PATH}.")
        return FALLBACK_RVC_MODEL_PATH
    raise FileNotFoundError(f'RVC model was not found: {requested_path}')


def find_index_path(applio_root: Path) -> Path | None:
    search_roots = [
        DEFAULT_RVC_MODEL_PATH.parent,
        applio_root / 'logs',
        applio_root / 'assets',
    ]
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(path for path in root.rglob('*.index') if path.is_file())
        if matches:
            return matches[0]
    return None


def write_wav_mono(path: Path, audio: torch.Tensor | np.ndarray, sample_rate: int) -> None:
    if isinstance(audio, torch.Tensor):
        audio_np = audio.detach().cpu().numpy()
    else:
        audio_np = np.asarray(audio)

    audio_np = np.squeeze(audio_np).astype(np.float32, copy=False)
    if audio_np.ndim != 1:
        raise ValueError(f'Silero returned audio with unsupported shape: {audio_np.shape}')

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_i16 = (audio_np * 32767.0).astype('<i2')

    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_i16.tobytes())


def read_wav_for_playback(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), 'rb') as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f'Only PCM16 WAV playback is supported, got sample_width={sample_width}.')

    audio = np.frombuffer(frames, dtype='<i2')
    if channels > 1:
        audio = audio.reshape(-1, channels)
    audio_float = audio.astype(np.float32) / 32768.0
    return audio_float, sample_rate


def play_wav(path: Path, output_device: int | str | None) -> None:
    audio, sample_rate = read_wav_for_playback(path)
    if audio.ndim == 1:
        audio = np.repeat(audio[:, None], 2, axis=1)
    sd.play(audio, samplerate=sample_rate, device=output_device, blocking=True)
    sd.stop()


def sanitize_filename_part(value: str) -> str:
    sanitized = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in value.strip())
    return sanitized or 'value'


def load_silero_tts(repo_path: Path, model_id: str, device: torch.device):
    source = 'local' if repo_path.exists() else 'github'
    repo = str(repo_path) if source == 'local' else 'snakers4/silero-models'
    model, _example_text = torch.hub.load(
        repo_or_dir=repo,
        model='silero_tts',
        language='ru',
        speaker=model_id,
        source=source,
        trust_repo=True,
    )
    model.to(device)
    return model


def synthesize_silero(
    *,
    text: str,
    output_path: Path,
    repo_path: Path,
    model_id: str,
    speaker: str,
    sample_rate: int,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = load_silero_tts(repo_path, model_id, device)
    audio = model.apply_tts(text=text, speaker=speaker, sample_rate=sample_rate)
    write_wav_mono(output_path, audio, sample_rate)


def run_applio_rvc(
    *,
    applio_root: Path,
    applio_python: Path,
    input_path: Path,
    output_path: Path,
    model_path: Path,
    index_path: Path | None,
    pitch: int,
    f0_method: str,
    index_rate: float,
    protect: float,
) -> None:
    effective_index_path = str(index_path) if index_path is not None else ''
    effective_index_rate = index_rate if index_path is not None else 0.0
    if index_path is None:
        print('RVC index was not found; conversion will run without index retrieval.')

    command = [
        str(applio_python),
        str(applio_root / 'core.py'),
        'infer',
        '--pitch',
        str(pitch),
        '--index_rate',
        str(effective_index_rate),
        '--volume_envelope',
        '1.0',
        '--protect',
        str(protect),
        '--f0_method',
        f0_method,
        '--input_path',
        str(input_path),
        '--output_path',
        str(output_path),
        '--pth_path',
        str(model_path),
        '--index_path',
        effective_index_path,
        '--split_audio',
        'False',
        '--f0_autotune',
        'False',
        '--clean_audio',
        'False',
        '--export_format',
        'WAV',
        '--embedder_model',
        'contentvec',
    ]

    subprocess.run(command, cwd=str(applio_root), check=True)
    if not output_path.exists():
        raise RuntimeError(f'Applio RVC did not create the expected output file: {output_path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Local Silero TTS -> Applio RVC -> playback test for Herta voice.',
    )
    parser.add_argument('text', nargs='?', help='Text to synthesize. If omitted, stdin is used.')
    parser.add_argument('--applio-root', type=Path, default=DEFAULT_APPLIO_ROOT)
    parser.add_argument('--applio-python', type=Path, default=DEFAULT_APPLIO_ROOT / 'env' / 'python.exe')
    parser.add_argument('--rvc-model', type=Path, default=DEFAULT_RVC_MODEL_PATH)
    parser.add_argument('--rvc-index', type=Path, default=None)
    parser.add_argument('--pitch', type=int, default=DEFAULT_PITCH)
    parser.add_argument(
        '--pitch-sweep',
        nargs='*',
        type=int,
        default=None,
        help='Generate several output files with different pitch values, for example: --pitch-sweep 0 3 5 7.',
    )
    parser.add_argument('--f0-method', default=DEFAULT_F0_METHOD)
    parser.add_argument('--index-rate', type=float, default=0.3)
    parser.add_argument('--protect', type=float, default=0.33)
    parser.add_argument('--silero-repo', type=Path, default=DEFAULT_SILERO_REPO)
    parser.add_argument('--silero-model', default='v4_ru')
    parser.add_argument('--silero-speaker', default='xenia')
    parser.add_argument('--silero-sample-rate', type=int, default=48000)
    parser.add_argument('--silero-device', default='cpu')
    parser.add_argument('--output-device', default=None)
    parser.add_argument('--output-wav', type=Path, default=None)
    parser.add_argument('--keep-temp', action='store_true')
    parser.add_argument('--no-play', action='store_true')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    text = args.text if args.text is not None else input('Text> ')
    text = text.strip()
    if not text:
        print('No text was provided.')
        return 1

    applio_root = args.applio_root.resolve()
    applio_python = args.applio_python.resolve()
    if not applio_python.exists():
        raise FileNotFoundError(f'Applio Python was not found: {applio_python}')

    model_path = resolve_rvc_model_path(args.rvc_model)
    index_path = args.rvc_index if args.rvc_index is not None else find_index_path(applio_root)
    if index_path is not None and not index_path.exists():
        raise FileNotFoundError(f'RVC index was not found: {index_path}')

    output_device = parse_device(args.output_device)

    with tempfile.TemporaryDirectory(prefix='herta_rvc_tts_') as temp_dir:
        temp_path = Path(temp_dir)
        base_wav = temp_path / 'silero_base.wav'
        print('Synthesizing base voice with Silero...')
        synthesize_silero(
            text=text,
            output_path=base_wav,
            repo_path=args.silero_repo,
            model_id=args.silero_model,
            speaker=args.silero_speaker,
            sample_rate=args.silero_sample_rate,
            device_name=args.silero_device,
        )

        pitch_values = args.pitch_sweep
        if pitch_values is not None and len(pitch_values) == 0:
            pitch_values = list(DEFAULT_PITCH_SWEEP)
        if pitch_values is None:
            pitch_values = [args.pitch]

        output_files: list[Path] = []
        for pitch in pitch_values:
            if len(pitch_values) == 1:
                rvc_wav = args.output_wav.resolve() if args.output_wav is not None else temp_path / 'herta_rvc.wav'
            else:
                output_dir = args.output_wav.resolve() if args.output_wav is not None else Path.cwd() / 'data' / 'pitch_sweep'
                if output_dir.suffix.lower() == '.wav':
                    output_dir = output_dir.parent
                rvc_wav = output_dir / f'herta_rvc_pitch_{sanitize_filename_part(str(pitch))}.wav'

            rvc_wav.parent.mkdir(parents=True, exist_ok=True)
            print(f'Converting base voice with Applio RVC... pitch={pitch}, f0_method={args.f0_method}')
            run_applio_rvc(
                applio_root=applio_root,
                applio_python=applio_python,
                input_path=base_wav,
                output_path=rvc_wav,
                model_path=model_path,
                index_path=index_path,
                pitch=pitch,
                f0_method=args.f0_method,
                index_rate=args.index_rate,
                protect=args.protect,
            )
            output_files.append(rvc_wav)
            print(f'Output written to: {rvc_wav}')

        if args.no_play:
            if len(output_files) > 1:
                print('Pitch sweep completed:')
                for path in output_files:
                    print(f'- {path}')
            elif not args.output_wav and not args.keep_temp:
                print('Use --output-wav to keep the generated file.')
            return 0

        for path in output_files:
            print(f'Playing: {path}')
            play_wav(path, output_device)

        if args.keep_temp and not args.output_wav and len(output_files) == 1:
            kept_path = Path.cwd() / 'herta_rvc_output.wav'
            kept_path.write_bytes(output_files[0].read_bytes())
            print(f'Output copied to: {kept_path}')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
