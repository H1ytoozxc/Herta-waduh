import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr


def send_message(payload: dict[str, object]) -> None:
    PROTOCOL_STDOUT.write(json.dumps(payload) + '\n')
    PROTOCOL_STDOUT.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Persistent Applio RVC worker.')
    parser.add_argument('--applio-root', type=Path, required=True)
    return parser


def import_voice_converter(applio_root: Path):
    os.chdir(applio_root)
    sys.path.insert(0, str(applio_root))
    from rvc.infer.infer import VoiceConverter

    return VoiceConverter()


def preload_model(converter, request: dict[str, object]) -> None:
    model_path = str(request['model_path'])
    embedder_model = str(request.get('embedder_model') or 'contentvec')
    converter.get_vc(model_path, int(request.get('sid') or 0))
    if converter.vc is None:
        raise RuntimeError(f'Applio RVC model was not loaded: {model_path}')
    if converter.hubert_model is None or converter.last_embedder_model != embedder_model:
        converter.load_hubert(embedder_model, request.get('embedder_model_custom'))
        converter.last_embedder_model = embedder_model


def convert_audio(converter, request: dict[str, object]) -> None:
    output_path = Path(str(request['output_path']))
    if output_path.exists():
        output_path.unlink()

    converter.convert_audio(
        audio_input_path=str(request['input_path']),
        audio_output_path=str(output_path),
        model_path=str(request['model_path']),
        index_path=str(request.get('index_path') or ''),
        pitch=int(request.get('pitch') or 0),
        f0_method=str(request.get('f0_method') or 'rmvpe'),
        index_rate=float(request.get('index_rate') or 0.0),
        volume_envelope=float(request.get('volume_envelope') or 1.0),
        protect=float(request.get('protect') or 0.33),
        split_audio=False,
        f0_autotune=False,
        f0_autotune_strength=1.0,
        proposed_pitch=False,
        proposed_pitch_threshold=155.0,
        clean_audio=False,
        clean_strength=0.5,
        export_format='WAV',
        embedder_model=str(request.get('embedder_model') or 'contentvec'),
        embedder_model_custom=request.get('embedder_model_custom'),
        post_process=False,
        sid=int(request.get('sid') or 0),
    )

    if not output_path.exists():
        raise RuntimeError(f'Applio RVC did not create output file: {output_path}')


def main() -> int:
    args = build_parser().parse_args()
    converter = import_voice_converter(args.applio_root.resolve())
    send_message({'status': 'ready'})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        started_at = time.perf_counter()
        try:
            request = json.loads(line)
            command = request.get('command')

            if command == 'shutdown':
                send_message({'status': 'ok', 'command': command})
                break
            if command == 'preload':
                preload_model(converter, request)
            elif command == 'convert':
                convert_audio(converter, request)
            else:
                raise ValueError(f'Unsupported worker command: {command}')

            send_message({
                'status': 'ok',
                'command': command,
                'elapsed_seconds': round(time.perf_counter() - started_at, 3),
            })
        except Exception as exc:
            send_message({
                'status': 'error',
                'error': str(exc),
                'traceback': traceback.format_exc(),
            })

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
