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
- STT через локальный `faster-whisper` или через Google AI Studio
- persona-слой для Великой Герты с более естественным разговорным режимом
- локальная память последних диалогов между перезапусками
- безопасные системные действия: открыть браузер/сайт/поиск, открыть VS Code, создать новый `.txt`
- голосовое взаимодействие на русском языке
- автоматическое определение языка Whisper для смешанного русского и английского ввода
- опциональная озвучка ответов через Edge TTS

Что пока не реализовано:
- активация по wake word
- произвольный tool calling или выполнение shell-команд от имени ассистента
- долгосрочная структурированная память профиля и фактов
- полноценная двуязычная стратегия диалога и автоматическое переключение TTS по языку

Иначе говоря: в текущей альфе ассистент умеет слушать, распознавать, думать, отвечать, при необходимости озвучивать ответ и выполнять небольшой набор безопасных системных действий. Полного доступа к системе у него нет.

## Стек

- Python
- Ollama
- локальная LLM: `qwen3:4b` по умолчанию
- опциональный облачный LLM-провайдер: DeepSeek API, OpenRouter или Google AI Studio
- `sounddevice` для аудиоввода
- `silero-vad` для сегментации речи
- `faster-whisper` для распознавания речи
- Google AI Studio для опционального облачного распознавания речи
- `edge-tts`, SAPI/Piper или Silero для базового синтеза речи
- Applio/RVC для опционального голоса Герты поверх базового TTS
- безопасный allowlist-слой для системных действий без произвольной консоли

## Структура проекта

```text
The_Herta_Voice_Assistant/
├─ main.py
├─ config.py
├─ audio/
├─ actions/
├─ stt/
├─ tts/
├─ llm/
├─ persona/
├─ utils/
├─ wakeword/
└─ brain/
```

Примечание: в репозитории все еще лежат некоторые ранние директории из первого каркаса. Текущий рабочий рантайм использует прежде всего `audio/`, `stt/`, `tts/`, `llm/`, `persona/`, `actions/` и `utils/`.

## Личность Герты

Герта настроена не как нейтральный сервисный ассистент, а как Великая Герта:
- 83-й член Общества гениев;
- Эманатор Эрудиции;
- высокомерный, лаконичный и интеллектуально-доминантный собеседник;
- сухой сарказм вместо дружелюбной нейтральности;
- уважение к чистому коду, строгой типизации, модульности, эффективности и элегантным алгоритмам.

В технических задачах persona-слой подталкивает модель оценивать:
- архитектуру и границы ответственности;
- сложность алгоритма и лишние проходы по данным;
- типизацию и валидацию;
- длину и чистоту функций;
- избыточность, канцелярит и плохие абстракции.

При этом характер не должен заменять пользу. Герта может быть язвительной, но после колкости должна дать рабочий технический ответ. Для Live-моделей используется компактный persona prompt, поэтому изменения личности применяются и в `--live-voice`.

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

Для голосового режима основной путь теперь Google Live API. Он обходит локальные Whisper/STT/TTS/RVC и использует нативное аудио Gemini напрямую.

Если нужен оригинальный RVC-голос Герты без лимитных `generateContent`/Google STT моделей, используй `--live-voice` вместе с `GOOGLE_AI_LIVE_PLAYBACK='rvc'`. В этом режиме Google Live остается единственным облачным AI/STT-движком, а локальный RVC озвучивает transcript ответа.

Рекомендуемый Live-режим: Gemini 3.1 Flash Live Preview.

```powershell
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_API_VERSION='v1beta'
$env:GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='false'
$env:GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:GOOGLE_AI_LIVE_PLAYBACK='google'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --live-voice
```

Запасной/альтернативный Live-режим: Gemini 2.5 Flash Native Audio Dialog.

```powershell
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_LIVE_MODEL='gemini-2.5-flash-native-audio-preview-12-2025'
$env:GOOGLE_AI_LIVE_API_VERSION='v1alpha'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='true'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:GOOGLE_AI_LIVE_PLAYBACK='google'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --live-voice
```

Обычный пайплайн `--voice` (`STT -> LLM -> TTS/RVC`) остается в проекте как запасной режим, но для живого голосового общения предпочтительнее использовать только `--live-voice` с одной из двух моделей выше.

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

## Установка из архива на Windows

Этот сценарий подходит, если у пользователя есть только архив проекта и VS Code. Git не обязателен.

Что нужно заранее:
- Windows 10/11.
- VS Code.
- Python 3.11 или новее. При установке Python обязательно включить галочку `Add python.exe to PATH`.
- Интернет для установки зависимостей и первой загрузки Whisper-модели.
- Микрофон и устройство вывода звука.
- Google AI Studio API key, если запускать через `LLM_PROVIDER='google_ai'`.

Рекомендуемый путь для распаковки архива:

```text
C:\Herta\The_Herta_Voice_Assistant
```

Лучше избегать слишком длинных путей, пробелов и кириллицы в пути к проекту. Это не всегда ломает Python, но сильно упрощает диагностику.

1. Распакуй архив.
2. Открой VS Code.
3. Нажми `File -> Open Folder`.
4. Выбери папку проекта, где лежит `main.py`.
5. Открой терминал VS Code: `Terminal -> New Terminal`.
6. Убедись, что терминал открыт в папке проекта:

```powershell
pwd
```

В выводе должен быть путь к папке `The_Herta_Voice_Assistant`.

Проверь Python:

```powershell
python --version
```

Если команда не найдена, попробуй:

```powershell
py --version
```

Если Python не найден вообще, установи Python заново и включи `Add python.exe to PATH`.

Создай виртуальное окружение:

```powershell
python -m venv .venv
```

Если работает только `py`, используй:

```powershell
py -3.11 -m venv .venv
```

Активируй окружение:

```powershell
.\.venv\Scripts\Activate.ps1
```

Если PowerShell ругается на execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
```

После активации слева в терминале должно появиться `(.venv)`.

Обнови pip и поставь зависимости:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Минимальная проверка без микрофона и без озвучки:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='сюда_вставить_ключ'
$env:GOOGLE_AI_MODEL='gemma-3-27b-it'
$env:GOOGLE_AI_MAX_TOKENS='220'
$env:GOOGLE_AI_TIMEOUT_SECONDS='45'

python main.py --text "Привет, кто ты?" --no-tts
```

Если Герта ответила текстом, LLM-часть работает.

Проверь аудиоустройства:

```powershell
python main.py --list-devices
python main.py --list-output-devices
```

Найди в списке индекс микрофона и индекс колонок/наушников. Например:

```text
[7] Микрофон ...
[9] Динамики ...
```

Затем проверь вывод звука:

```powershell
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --output-test
```

Если слышен короткий тон, вывод звука настроен.

Проверь TTS без RVC:

```powershell
$env:AUDIO_OUTPUT_DEVICE='9'
$env:RVC_TTS_ENABLED='false'
python main.py --tts-test
```

Если голос прозвучал, обычная озвучка работает.

Основной голосовой запуск через Gemini 3.1 Flash Live:

```powershell
$env:GOOGLE_AI_API_KEY='сюда_вставить_ключ'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_API_VERSION='v1beta'
$env:GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='false'
$env:GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:GOOGLE_AI_LIVE_PLAYBACK='google'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'

python main.py --live-voice
```

Когда появится строка `Voice mode ready`, можно говорить в микрофон. Остановить ассистента можно через `Ctrl+C`.

Если Gemini 3.1 Flash Live в аккаунте недоступна или работает нестабильно, используй Gemini 2.5 Flash Native Audio Dialog:

```powershell
$env:GOOGLE_AI_API_KEY='сюда_вставить_ключ'
$env:GOOGLE_AI_LIVE_MODEL='gemini-2.5-flash-native-audio-preview-12-2025'
$env:GOOGLE_AI_LIVE_API_VERSION='v1alpha'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='true'
$env:GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:GOOGLE_AI_LIVE_PLAYBACK='google'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'

python main.py --live-voice
```

По умолчанию `--live-voice` не использует локальный Whisper, Google STT, Edge/SAPI/Piper, Silero или RVC. Микрофон отправляется в Live API, ответ приходит нативным голосом Gemini.

Запуск только через Live-модель, но с оригинальным RVC-голосом Герты:

```powershell
$env:GOOGLE_AI_API_KEY='сюда_вставить_ключ'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_API_VERSION='v1beta'
$env:GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'
$env:GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='false'
$env:GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'
$env:GOOGLE_AI_LIVE_INPUT_TRANSCRIPTION='true'
$env:GOOGLE_AI_LIVE_OUTPUT_TRANSCRIPTION='true'
$env:GOOGLE_AI_LIVE_PLAYBACK='rvc'

$env:RVC_TTS_ENABLED='true'
$env:RVC_BACKEND='persistent'
$env:RVC_WARM_UP='true'
$env:RVC_BASE_TTS='silero'
$env:RVC_APPLIO_ROOT='Z:\APPLIO'
$env:RVC_APPLIO_PYTHON='Z:\APPLIO\env\python.exe'
$env:RVC_MODEL_PATH='Z:\ГЕРТАААА\model.pth'
$env:RVC_INDEX_PATH=''
$env:RVC_PITCH='0'
$env:RVC_F0_METHOD='rmvpe'

$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'

python main.py --live-voice
```

В этом режиме Live API все равно генерирует native audio, потому что native audio-модели требуют `AUDIO` response modality, но проект не проигрывает гугловский звук. Он ждет transcript ответа и прогоняет его через локальный RVC.

Если нужен полностью локальный запасной режим без Live API:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='сюда_вставить_ключ'
$env:GOOGLE_AI_MODEL='gemma-3-27b-it'
$env:STT_PROVIDER='whisper'
$env:RVC_TTS_ENABLED='false'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
$env:WHISPER_MODEL_SIZE='small'
$env:WHISPER_DEVICE='cpu'

python main.py --voice
```

На первом запуске Whisper может скачать модель. Это нормально и может занять несколько минут. Этот режим нужен как fallback, а не как основной голосовой путь.

Чтобы не вводить переменные окружения каждый раз, можно создать файл `.env` в корне проекта:

```env
GOOGLE_AI_API_KEY=сюда_вставить_ключ
GOOGLE_AI_LIVE_MODEL=gemini-3.1-flash-live-preview
GOOGLE_AI_LIVE_API_VERSION=v1beta
GOOGLE_AI_LIVE_THINKING_LEVEL=minimal
GOOGLE_AI_LIVE_AFFECTIVE_DIALOG=false
GOOGLE_AI_LIVE_PROACTIVE_AUDIO=false
GOOGLE_AI_LIVE_VOICE=Kore
GOOGLE_AI_LIVE_PLAYBACK=google

RVC_TTS_ENABLED=false
AUDIO_DEVICE=7
AUDIO_OUTPUT_DEVICE=9

SYSTEM_ACTIONS_ENABLED=false
SYSTEM_ACTIONS_DOCUMENT_DIR=desktop
SYSTEM_ACTIONS_REGISTRY_PATH=data/system_actions_registry.json
```

Файл `.env` уже добавлен в `.gitignore`. Его нельзя отправлять в публичный репозиторий, потому что там лежит API-ключ.

После создания `.env` запуск короче:

```powershell
python main.py --live-voice
```

### Безопасные системные действия

Системные действия выключены по умолчанию. Чтобы Герта могла открывать браузер, VS Code, создавать папки и работать с текстовыми документами, включи allowlist:

```powershell
$env:SYSTEM_ACTIONS_ENABLED='true'
$env:SYSTEM_ACTIONS_DOCUMENT_DIR='desktop'
$env:SYSTEM_ACTIONS_REGISTRY_PATH='data/system_actions_registry.json'
$env:SYSTEM_ACTIONS_BROWSER_HOME_URL='https://www.google.com'
$env:SYSTEM_ACTIONS_VSCODE_COMMAND='code'
$env:SYSTEM_ACTIONS_VSCODE_OPEN_WORKSPACE='true'
```

Разрешено:
- `открой браузер`;
- `открой google.com`;
- `загугли погоду в Москве`;
- `открой VS Code`;
- `создай папку проект Герты`;
- `создай папку и назови ее как-нибудь`;
- `создай текстовый документ`;
- `создай текстовый документ с названием план и текстом купить чай`;
- `создай папку материалы и документ план с текстом первая строка`;
- `допиши в документ план текст купить молоко`;
- `добавь позвонить завтра в документ план`;
- `допиши в документ план текст вторая строка в папке материалы`;
- `переименуй папку проект Герты в архив Герты`;
- `переименуй документ план в задачи`.

Запрещено всегда:
- удалять файлы;
- перемещать файлы;
- перезаписывать существующие файлы;
- форматировать диски;
- выполнять произвольные команды PowerShell/CMD.

Текстовые документы и папки создаются только внутри `SYSTEM_ACTIONS_DOCUMENT_DIR`. Значение `desktop` означает рабочий стол Windows. Если пользователь просит назвать объект "как-нибудь", Герта сама генерирует имя вместо буквального `как-нибудь`.

Файлы можно только дописывать в конец `.txt`. Переименование разрешено только для файлов и папок, которые сама Герта создала и записала в `SYSTEM_ACTIONS_REGISTRY_PATH`. Чужие файлы на рабочем столе она не переименовывает и не меняет.

### Опционально: RVC-голос Герты

RVC не входит в обычную установку из архива. Его нужно ставить отдельно.

Пользователю нужны:
- установленный Applio;
- `.pth` модель голоса;
- желательно `.index` файл модели, но можно запускать и без него;
- достаточно мощный ПК, потому что RVC-конвертация заметно тормозит ответы.

Пример путей:

```text
C:\Applio
C:\HertaVoice\model.pth
C:\HertaVoice\model.index
```

Проверка Applio Python:

```powershell
C:\Applio\env\python.exe --version
```

Переменные для RVC:

```powershell
$env:RVC_TTS_ENABLED='true'
$env:RVC_BACKEND='persistent'
$env:RVC_WARM_UP='true'
$env:RVC_BASE_TTS='silero'
$env:RVC_APPLIO_ROOT='C:\Applio'
$env:RVC_APPLIO_PYTHON='C:\Applio\env\python.exe'
$env:RVC_MODEL_PATH='C:\HertaVoice\model.pth'
$env:RVC_INDEX_PATH='C:\HertaVoice\model.index'
$env:RVC_PITCH='0'
$env:RVC_F0_METHOD='rmvpe'
$env:SILERO_TTS_SAMPLE_RATE='24000'
$env:AUDIO_OUTPUT_DEVICE='9'

python main.py --tts-test
```

Если `.index` файла нет:

```powershell
$env:RVC_INDEX_PATH=''
```

После успешного `--tts-test` можно запускать обычный голосовой режим:

```powershell
python main.py --voice
```

### Частые ошибки

`ModuleNotFoundError`:
активируй `.venv` и повтори установку зависимостей.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`GOOGLE_AI_API_KEY is not configured`:
ключ не задан. Проверь `$env:GOOGLE_AI_API_KEY` или файл `.env`.

`401 Unauthorized`:
ключ неправильный, отключен или вставлен с лишними пробелами.

`429 Too Many Requests`:
лимит провайдера. Нужно подождать, уменьшить `GOOGLE_AI_MAX_TOKENS` или сменить модель.

Whisper долго загружается:
на первом запуске это нормально. Он скачивает модель распознавания речи.

Нет звука:
проверь `python main.py --list-output-devices`, выставь правильный `AUDIO_OUTPUT_DEVICE` и запусти `python main.py --output-test`.

Ассистент не слышит микрофон:
проверь `python main.py --list-devices`, выставь правильный `AUDIO_DEVICE`, а также разрешения микрофона в Windows.

RVC очень медленный:
это ожидаемо. `RVC_BACKEND='persistent'` убирает повторную загрузку модели между ответами, но сама конвертация каждого аудиофайла все равно занимает время.

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

Основной голосовой режим через Gemini 3.1 Flash Live:

```powershell
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'
$env:GOOGLE_AI_LIVE_API_VERSION='v1beta'
$env:GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'
$env:GOOGLE_AI_LIVE_VOICE='Kore'
$env:AUDIO_DEVICE='7'
$env:AUDIO_OUTPUT_DEVICE='9'
python main.py --live-voice
```

Альтернативный голосовой режим через Gemini 2.5 Flash Native Audio Dialog:

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

Запасной обычный пайплайн через Google STT и RVC-голос Герты:

```powershell
$env:LLM_PROVIDER='google_ai'
$env:GOOGLE_AI_API_KEY='AIza...'
$env:GOOGLE_AI_MODEL='gemma-3-27b-it'
$env:GOOGLE_AI_MAX_TOKENS='220'
$env:GOOGLE_AI_TIMEOUT_SECONDS='45'

$env:STT_PROVIDER='whisper'

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

Этот режим оставлен как fallback. Для живого голосового общения предпочтительнее `--live-voice` с `gemini-3.1-flash-live-preview` или `gemini-2.5-flash-native-audio-preview-12-2025`.

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
$env:GOOGLE_AI_FALLBACK_MODEL=''
$env:GOOGLE_AI_MAX_TOKENS='700'
$env:GOOGLE_AI_TIMEOUT_SECONDS='45'
$env:GOOGLE_AI_RETRY_ATTEMPTS='0'
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
$env:GOOGLE_AI_LIVE_PLAYBACK='google'
$env:STT_PROVIDER='whisper'
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
$env:SYSTEM_ACTIONS_ENABLED='false'
$env:SYSTEM_ACTIONS_DOCUMENT_DIR='desktop'
$env:SYSTEM_ACTIONS_REGISTRY_PATH='data/system_actions_registry.json'
$env:SYSTEM_ACTIONS_BROWSER_HOME_URL='https://www.google.com'
$env:SYSTEM_ACTIONS_VSCODE_COMMAND='code'
$env:SYSTEM_ACTIONS_VSCODE_OPEN_WORKSPACE='true'
```

Примечания:
- Если `WHISPER_LANGUAGE` не задан, Whisper сам определяет язык.
- Для смешанного русского и английского ввода лучше оставить `WHISPER_LANGUAGE` пустым.
- Если принудительно выставить `WHISPER_LANGUAGE='ru'`, качество распознавания английского снизится.
- `DEEPSEEK_RATE_LIMIT_RETRIES` задает число повторов после `429 Too Many Requests`.
- Для голосового режима с бесплатными OpenRouter-моделями обычно удобнее держать `DEEPSEEK_RATE_LIMIT_RETRIES='2'`, чтобы ассистент не зависал надолго в ожидании лимита.
- `GOOGLE_AI_MODEL`, `GOOGLE_AI_FALLBACK_MODEL`, `GOOGLE_AI_TIMEOUT_SECONDS` и `GOOGLE_AI_RETRY_ATTEMPTS` относятся к обычному текстовому/legacy `--voice` пайплайну. Они не используются в `--live-voice`.
- `GOOGLE_AI_RATE_LIMIT_RETRIES` задает число повторов после лимитов Google AI Studio.
- `GOOGLE_AI_LIVE_MODEL='gemini-3.1-flash-live-preview'` включает основной Live API режим с Gemini 3.1 Flash Live Preview.
- `GOOGLE_AI_LIVE_MODEL='gemini-2.5-flash-native-audio-preview-12-2025'` включает альтернативный Live API режим Gemini 2.5 Flash Native Audio Dialog.
- Для Gemini 3.1 Flash Live используй `GOOGLE_AI_LIVE_API_VERSION='v1beta'`, `GOOGLE_AI_LIVE_THINKING_LEVEL='minimal'`, `GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='false'` и `GOOGLE_AI_LIVE_PROACTIVE_AUDIO='false'`.
- Для Gemini 2.5 Flash Native Audio Dialog можно использовать `GOOGLE_AI_LIVE_API_VERSION='v1alpha'` и `GOOGLE_AI_LIVE_AFFECTIVE_DIALOG='true'`.
- `GOOGLE_AI_LIVE_VOICE='Kore'` фиксирует голос Gemini Live, чтобы он не менялся между ответами. В режиме `GOOGLE_AI_LIVE_PLAYBACK='rvc'` этот голос генерируется Google, но не проигрывается.
- `GOOGLE_AI_LIVE_PLAYBACK='google'` проигрывает нативное аудио Gemini Live напрямую.
- `GOOGLE_AI_LIVE_PLAYBACK='rvc'` игнорирует нативное аудио Gemini Live, берет `output_audio_transcription` и озвучивает текст через локальный RVC-голос Герты.
- `--live-voice` не использует локальные Whisper и Google STT: распознавание речи делает сама Live-модель.
- `STT_PROVIDER`, `GOOGLE_STT_MODEL` и `WHISPER_*` относятся только к legacy `--voice`. В `--live-voice` распознавание речи делает сама Live-модель.
- `RVC_TTS_ENABLED='true'` включает локальную цепочку `Silero TTS -> Applio RVC -> playback` для обычного режима `--voice` и для `--live-voice`, если задано `GOOGLE_AI_LIVE_PLAYBACK='rvc'`.
- `RVC_BACKEND='persistent'` держит Applio/RVC-процесс живым между ответами, чтобы не запускать тяжелую конвертацию с нуля каждый раз.
- `RVC_WARM_UP='true'` заранее загружает базовый TTS, RVC-модель и embedder при старте, поэтому первая реплика после запуска меньше тормозит.
- `RVC_BASE_TTS='silero'` использует Silero как базовый голос перед RVC. Можно попробовать `RVC_BASE_TTS='piper'`, но на текущем тесте он не оказался быстрее по общей задержке.
- `RVC_PITCH='0'` оставляет тональность RVC без повышения; `RVC_F0_METHOD='rmvpe'` использует RMVPE.
- `SYSTEM_ACTIONS_ENABLED='true'` включает безопасный allowlist действий: браузер, VS Code, папки, `.txt`, дописывание и ограниченное переименование.
- `SYSTEM_ACTIONS_DOCUMENT_DIR` задает папку для новых папок и текстовых документов. `desktop` кладет файлы на рабочий стол. Путь из голосовой команды не принимается, чтобы ассистент не писал куда попало.
- `SYSTEM_ACTIONS_REGISTRY_PATH` хранит список файлов и папок, созданных Гертой. Переименовывать можно только объекты из этого списка.
- Удаление, перемещение, перезапись файлов и произвольные shell-команды заблокированы на уровне кода, а не только промпта.
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
git add README.md config.py main.py persona/the_herta.py actions/__init__.py actions/system_actions.py llm/google_ai_client.py llm/google_live_client.py stt/google_ai_stt.py

git diff --cached --name-only
git commit -m "Add Gemini Live voice, safe OS actions, and Herta persona"
```

Если в `git diff --cached --name-only` видны `data/`, `.env`, `.wav`, `.mp3` или локальные модели, остановись и убери их из индекса:

```powershell
git restore --staged data .env .env.local
```

## Текущая модель взаимодействия

Пайплайн:

```text
Live voice:
microphone -> Gemini Live -> native audio

Live voice with RVC:
microphone -> Gemini Live -> output transcript -> Silero/RVC -> playback

Legacy voice:
microphone -> VAD -> STT -> LLM -> TTS/RVC -> playback
```

Текущее поведение:
- Ассистент умеет поддерживать диалог и отвечать на вопросы.
- Ассистент сохраняет последние реплики в `data/dialogue_memory.json` и подмешивает их в контекст после перезапуска.
- Герта умеет открывать браузер, запускать VS Code, создавать папки, создавать и дописывать `.txt`, а также переименовывать только те файлы и папки, которые создала сама.
- Удаление, перемещение, перезапись и произвольные команды PowerShell/CMD заблокированы.
- Полный tool/agent-слой пока не реализован: системные действия работают через локальный allowlist, а не через свободное выполнение команд моделью.

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



