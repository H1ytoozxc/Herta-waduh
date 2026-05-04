# The Herta Voice Assistant

Альфа-версия локального голосового ассистента в образе Великой Герты из Honkai: Star Rail.

Текущий фокус:
- локальный пайплайн без лишней инфраструктуры
- простая модульная архитектура на Python
- голосовое общение с персонажной подачей
- минимальная и понятная база для соло-разработки под Windows

## Статус альфы

Что уже работает:
- текстовый чат через локальный Ollama, DeepSeek/OpenRouter или Google AI Studio
- голосовой режим: микрофон -> VAD -> STT -> LLM -> опциональный TTS
- persona-слой для Великой Герты с более естественным разговорным режимом
- локальная память последних диалогов между перезапусками
- голосовое взаимодействие на русском языке
- автоматическое определение языка Whisper для смешанного русского и английского ввода
- опциональная озвучка ответов через Edge TTS

Что пока не реализовано:
- активация по wake word
- прямое управление Windows, файлами, приложениями, браузером или системными действиями
- tool calling или выполнение команд от имени ассистента
- долгосрочная структурированная память профиля и фактов
- полноценная двуязычная стратегия диалога и автоматическое переключение TTS по языку

Иначе говоря: в текущей альфе ассистент умеет слушать, распознавать, думать, отвечать и при необходимости озвучивать ответ. Управлять компьютером он пока не умеет.

## Стек

- Python
- Ollama
- локальная LLM: `qwen3:4b` по умолчанию
- опциональный облачный LLM-провайдер: DeepSeek API, OpenRouter или Google AI Studio
- `sounddevice` для аудиоввода
- `silero-vad` для сегментации речи
- `faster-whisper` для распознавания речи
- `edge-tts`, SAPI/Piper или Silero для базового синтеза речи
- Applio/RVC для опционального голоса Герты поверх базового TTS

## Структура проекта

```text
The_Herta_Voice_Assistant/
├─ main.py
├─ config.py
├─ audio/
├─ stt/
├─ tts/
├─ llm/
├─ persona/
├─ utils/
├─ wakeword/
└─ brain/
```

Примечание: в репозитории все еще лежат некоторые ранние директории из первого каркаса. Текущий рабочий рантайм использует прежде всего `audio/`, `stt/`, `tts/`, `llm/`, `persona/` и `utils/`.

## Требования

- Windows
- Python 3.11+
- локально установленный Ollama
- запущенный Ollama-сервер на `http://127.0.0.1:11434`
- хотя бы одна загруженная модель в Ollama
- опционально: DeepSeek/OpenRouter API key, если используешь `LLM_PROVIDER='deepseek'`
- опционально: Google AI Studio API key, если используешь `LLM_PROVIDER='google_ai'`

Рекомендуемая модель:

```powershell
ollama pull qwen3:4b
```

Если нужно временно использовать другую установленную модель:

```powershell
$env:OLLAMA_MODEL='gemma3:4b'
```

Если нужно использовать DeepSeek API вместо локального Ollama:

```powershell
$env:LLM_PROVIDER='deepseek'
$env:DEEPSEEK_API_KEY='sk-...'
$env:DEEPSEEK_MODEL='deepseek-v4-flash'
```

Для совместимости можно также указать `DEEPSEEK_MODEL='deepseek-chat'`, но для новой настройки предпочтительнее `deepseek-v4-flash`.

Если ключ начинается с `sk-or-v1-`, это ключ OpenRouter, а не прямой ключ DeepSeek. Тогда нужно указать OpenRouter endpoint и OpenRouter-имя модели:

```powershell
$env:LLM_PROVIDER='deepseek'
$env:DEEPSEEK_BASE_URL='https://openrouter.ai/api/v1'
$env:DEEPSEEK_API_KEY='sk-or-v1-...'
$env:DEEPSEEK_MODEL='deepseek/deepseek-v3.2'
```

Для бесплатной Gemma 4 26B A4B в OpenRouter:

```powershell
$env:LLM_PROVIDER='deepseek'
$env:DEEPSEEK_BASE_URL='https://openrouter.ai/api/v1'
$env:DEEPSEEK_API_KEY='sk-or-v1-...'
$env:DEEPSEEK_MODEL='google/gemma-4-26b-a4b-it:free'
```

Если OpenRouter возвращает `429 Too Many Requests`, это лимит или очередь у провайдера, особенно частая история на бесплатных моделях. Можно подождать, сменить модель или убрать суффикс `:free`, если на OpenRouter доступна платная версия:

```powershell
$env:DEEPSEEK_MODEL='google/gemma-4-26b-a4b-it'
```

Если нужно использовать Gemma 3 27B через Google AI Studio:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_MODEL='gemma-3-27b-it'
```

Вместо `GOOGLE_AI_API_KEY` можно использовать `GEMINI_API_KEY`.
Для Gemma 3 отдельное поле system/developer instruction отключено на стороне Google API, поэтому по умолчанию persona-инструкции передаются внутри обычного текста запроса.

Если нужно использовать Gemini 3 Flash в обычном пайплайне `Whisper -> LLM -> TTS`, достаточно сменить текстовую модель:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_MODEL='gemini-3-flash-preview'
```

Если нужно использовать Google Live API с нативным аудио Gemini, запускай отдельный режим. Он обходит локальные Whisper и TTS:

```powershell
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --live-voice
```

Для Gemini 2.5 Flash Native Audio Dialog:

```powershell
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_LIVE_MODEL='gemini-2.5-flash-native-audio-preview-12-2025'
$env:GOOGLE_AI_LIVE_API_VERSION='v1alpha'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='true'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --live-voice
```

## Установка

Создай и активируй виртуальное окружение, затем установи зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если PowerShell блокирует активацию:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

## Быстрый старт

Текстовый режим:

```powershell
python main.py --text "Привет, кто ты?" --no-tts
```

Интерактивный текстовый режим:

```powershell
python main.py --no-tts
```

Список аудиоустройств:

```powershell
python main.py --list-devices
```

Голосовой режим без TTS:

```powershell
$env:AUDIO_DEVICE='7'
$env:WHISPER_MODEL_SIZE='small'
$env:WHISPER_DEVICE='cpu'
python main.py --voice --no-tts
```

Голосовой режим с TTS:

```powershell
$env:AUDIO_DEVICE='7'
$env:WHISPER_MODEL_SIZE='small'
$env:WHISPER_DEVICE='cpu'
python main.py --voice
```

Голосовой режим через Google AI Studio и RVC-голос Герты:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_MODEL='gemini-3-flash-preview'
$env:GOOGLE_AI_MAX_TOKENS='220'

$env:RVC_TTS_ENABLED='true'
$env:RVC_BACKEND='persistent'
$env:RVC_WARM_UP='true'
$env:RVC_BASE_TTS='silero'
$env:RVC_MODEL_PATH='Z:\ГЕРТАААА\model.pth'
$env:RVC_PITCH='0'
$env:RVC_F0_METHOD='rmvpe'
$env:SILERO_TTS_SAMPLE_RATE='24000'

$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
$env:WHISPER_MODEL_SIZE='small'
$env:WHISPER_DEVICE='cpu'

python main.py --voice
```

Быстрая проверка только озвучки Герты:

```powershell
$env:RVC_TTS_ENABLED='true'
$env:RVC_BACKEND='persistent'
$env:RVC_WARM_UP='true'
$env:RVC_MODEL_PATH='Z:\ГЕРТАААА\model.pth'
$env:RVC_PITCH='0'
$env:RVC_F0_METHOD='rmvpe'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --tts-test
```

## Полезные переменные окружения

```powershell
$env:OLLAMA_MODEL='qwen3:4b'
$env:OLLAMA_TEMPERATURE='0.55'
$env:OLLAMA_NUM_CTX='2048'
$env:OLLAMA_NUM_GPU='16'
$env:LLM_PROVIDER='ollama'
$env:DEEPSEEK_API_KEY='sk-...'
$env:DEEPSEEK_MODEL='deepseek-v4-flash'
$env:DEEPSEEK_MAX_TOKENS='700'
$env:DEEPSEEK_RETRY_ATTEMPTS='4'
$env:DEEPSEEK_RATE_LIMIT_RETRIES='2'
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_MODEL='gemma-3-27b-it'
$env:GOOGLE_AI_MAX_TOKENS='700'
$env:GOOGLE_AI_RETRY_ATTEMPTS='4'
$env:GOOGLE_AI_RATE_LIMIT_RETRIES='2'
$env:GOOGLE_AI_SYSTEM_INSTRUCTION_ENABLED='false'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_API_VERSION='v1beta'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'
$env:GOOGLE_AI_LIVE_THINKING_BUDGET=''
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='false'
$env:GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'
$env:GOOGLE_AI_LIVE_INPUT_TRANSCRIPTION='true'
$env:GOOGLE_AI_LIVE_OUTPUT_TRANSCRIPTION='true'
$env:RVC_TTS_ENABLED='false'
$env:RVC_BACKEND='persistent'
$env:RVC_WARM_UP='true'
$env:RVC_BASE_TTS='silero'
$env:RVC_MODEL_PATH='Z:\ГЕРТАААА\model.pth'
$env:RVC_INDEX_PATH=''
$env:RVC_PITCH='0'
$env:RVC_F0_METHOD='rmvpe'
$env:SILERO_TTS_MODEL='v4_ru'
$env:SILERO_TTS_SPEAKER='xenia'
$env:SILERO_TTS_SAMPLE_RATE='24000'
$env:AUDIO_DEVICE='7'
$env:WHISPER_MODEL_SIZE='small'
$env:WHISPER_DEVICE='cpu'
$env:WHISPER_LANGUAGE='ru'
$env:PERSONA_REWRITE_ENABLED='false'
$env:MEMORY_ENABLED='true'
$env:MEMORY_PATH='data/dialogue_memory.json'
$env:MEMORY_CONTEXT_MESSAGES='12'
$env:MEMORY_MAX_MESSAGES='80'
```

Примечания:
- Если `WHISPER_LANGUAGE` не задан, Whisper сам определяет язык.
- Для смешанного русского и английского ввода лучше оставить `WHISPER_LANGUAGE` пустым.
- Если принудительно выставить `WHISPER_LANGUAGE='ru'`, качество распознавания английского снизится.
- `DEEPSEEK_RATE_LIMIT_RETRIES` задает число повторов после `429 Too Many Requests`.
- Для голосового режима с бесплатными OpenRouter-моделями обычно удобнее держать `DEEPSEEK_RATE_LIMIT_RETRIES='2'`, чтобы ассистент не зависал надолго в ожидании лимита.
- `GOOGLE_AI_MODEL='gemma-3-27b-it'` включает Gemma 3 27B через Google AI Studio.
- `GOOGLE_AI_RATE_LIMIT_RETRIES` задает число повторов после лимитов Google AI Studio.
- `GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'` включает Gemini Live API в режиме `--live-voice`.
- `GOOGLE_AI_LIVE_MODEL='gemini-2.5-flash-native-audio-preview-12-2025'` включает Gemini 2.5 Flash Native Audio Dialog.
- `GOOGLE_AI_LIVE_VOICE='Kore'` фиксирует голос Gemini Live, чтобы он не менялся между ответами. Другие варианты можно послушать в Google AI Studio.
- `--live-voice` не использует локальные Whisper и TTS: аудио идет напрямую в Gemini Live, а ответ приходит нативным голосом 24 kHz PCM.
- `RVC_TTS_ENABLED='true'` включает локальную цепочку `Silero TTS -> Applio RVC -> playback` для обычного режима `--voice`.
- `RVC_BACKEND='persistent'` держит Applio/RVC-процесс живым между ответами, чтобы не запускать тяжелую конвертацию с нуля каждый раз.
- `RVC_WARM_UP='true'` заранее загружает базовый TTS, RVC-модель и embedder при старте, поэтому первая реплика после запуска меньше тормозит.
- `RVC_BASE_TTS='silero'` использует Silero как базовый голос перед RVC. Можно попробовать `RVC_BASE_TTS='piper'`, но на текущем тесте он не оказался быстрее по общей задержке.
- `RVC_PITCH='0'` оставляет тональность RVC без повышения; `RVC_F0_METHOD='rmvpe'` использует RMVPE.
- В текущей версии Applio `pm` не подходит для этого пайплайна: pipeline поддерживает `rmvpe`, `fcpe`, `crepe` и `crepe-tiny`.
- Если RVC почти не грузит CPU/GPU и в основном ест RAM, проверь устройство внутри Applio/RVC. Проектный `.venv` может быть CPU-only, но сама RVC-конвертация идет через `Z:\APPLIO\env\python.exe`; важна именно эта среда.
- Быстрая проверка CUDA в Applio:

```powershell
Push-Location Z:\APPLIO
.\env\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
Pop-Location
```

- В диспетчере задач Windows GPU-нагрузка может быть не на графике `3D`, а на `CUDA` или `Compute`. Для коротких фраз всплеск может быть коротким и легко пропускаться.
- Даже с `RVC_BACKEND='persistent'` RVC остается самым медленным этапом: модель и embedder грузятся один раз при старте, но каждый ответ все равно конвертируется отдельным аудиофайлом.
- `MEMORY_CONTEXT_MESSAGES` задает, сколько последних сообщений из памяти попадет в контекст при старте.
- `MEMORY_MAX_MESSAGES` задает, сколько сообщений хранится на диске.
- Чтобы очистить память диалогов: `Remove-Item data\dialogue_memory.json`.
- Если включен облачный LLM-провайдер, например OpenRouter/DeepSeek/Google AI Studio, загруженная из памяти история отправляется этому провайдеру как часть контекста.

## Приватные данные и коммиты

Не коммить реальные API-ключи, `.env`, аудио-артефакты и память диалогов. В `.gitignore` уже добавлены:

```text
.env
.env.*
data/
*.wav
*.mp3
models/
```

Безопасная проверка перед коммитом:

```powershell
git status --short
git check-ignore -v .env data\dialogue_memory.json data\herta_rvc_test.wav
rg -n --hidden --glob '!venv/**' --glob '!.venv/**' --glob '!data/**' "sk-or-v1-|AIza|DEEPSEEK_API_KEY|GOOGLE_AI_API_KEY|GEMINI_API_KEY" .
```

В выводе `rg` допустимы только плейсхолдеры вроде `sk-...`, `sk-or-v1-...`, `AIza...` и имена переменных окружения. Реального длинного ключа в коммите быть не должно.

Точечный `git add` для текущей версии, без `data/` и `.env`:

```powershell
git add .gitignore README.md requirements.txt main.py config.py `
  audio/output.py audio/live.py `
  brain/memory.py `
  llm/ollama_client.py llm/deepseek_client.py llm/google_ai_client.py llm/google_live_client.py llm/qwen3_direct.Modelfile `
  persona/the_herta.py `
  tts/edge_tts_engine.py tts/rvc_tts_engine.py `
  tools/applio_rvc_worker.py tools/herta_rvc_tts.py

git diff --cached --name-only
git commit -m "Add Google AI providers, dialogue memory, and RVC voice"
```

Если в `git diff --cached --name-only` видны `data/`, `.env`, `.wav`, `.mp3` или локальные модели, остановись и убери их из индекса:

```powershell
git restore --staged data .env .env.local
```

## Текущая модель взаимодействия

Пайплайн:

```text
microphone -> VAD -> STT -> LLM -> TTS -> playback
```

Текущее поведение:
- Ассистент умеет поддерживать диалог и отвечать на вопросы.
- Ассистент сохраняет последние реплики в `data/dialogue_memory.json` и подмешивает их в контекст после перезапуска.
- Ассистент пока не открывает программы, не нажимает кнопки, не управляет файлами, не ходит в браузер и не выполняет локальные команды от твоего имени.
- Tool/agent-слой пока не реализован. LLM сейчас только генерирует текстовые ответы.

## Известные ограничения

- `gemma3:4b` держит персонаж хуже, чем `qwen3:4b`
- распознавание английского стало лучше за счет автоопределения, но смешанный голосовой ввод все еще требует дополнительной проверки
- TTS сейчас использует один настроенный голос и не переключает язык автоматически
- модуль wake word существует только как заглушка
- долгосрочной памяти и профиля пользователя пока нет

## Рекомендуемые следующие шаги

1. Добавить поддержку wake word.
2. Добавить action/tool layer для безопасного локального взаимодействия с компьютером.
3. Четче разделить разговорный режим и task mode.
4. Добавить память с жесткими границами использования.
5. Улучшить двуязычное поведение STT и TTS.

## Цель альфы

Разумная цель для `v0.1-alpha` такая:
- стабильный текстовый режим
- стабильный голосовой режим
- приемлемое удержание персонажа
- никаких ложных заявлений о системных действиях
- задокументированная установка и ограничения

В этом направлении проект сейчас и движется.

## Примечание
Следите за апдейтами в моём тгк: https://t.me/cmd_phaeton_oq


<img width="373" height="224" alt="the-herta-hsr" src="https://github.com/user-attachments/assets/76d32225-e063-48c4-bae5-839d4ccb246f" />



