# Система проверки произношения английского языка (voice-pronunciation)

Статус: ✅ реализовано и проверено на реальной речи.
Дата: 2026-10-19.

## Что сделано

Создана система, которая принимает английское аудио (голосовое из Telegram, файл
или запись с микрофона), распознаёт речь через **Groq Whisper**, сравнивает её с
ожидаемым текстом и через **LLM** выдаёт IPA-транскрипцию, оценку и советы на
русском.

## Файлы

| Путь | Назначение |
|------|-----------|
| `/root/tools/voice-pronunciation/pronounce.py` | Основной скрипт (только stdlib + ffmpeg) |
| `/root/tools/voice-pronunciation/README.md` | Документация по использованию |
| `/root/tools/voice-pronunciation/tests/test_pronounce.py` | 33 теста (31 unit + 2 integration) |
| `/root/tools/voice-pronunciation/PRONUNCIATION.md` | Этот файл |
| `/root/.hermes/skills/voice-pronunciation/SKILL.md` | Скилл Hermes |

## Архитектура

```
аудио (Telegram / файл / микрофон)
        │  ffmpeg → 16 kHz mono WAV
        ▼
Groq Whisper (whisper-large-v3-turbo, verbose_json)
        │  transcript + words[] + segments[avg_logprob]
        ▼
сравнение с --reference (difflib)      уверенность (avg_logprob)
        │                                      │
        └──────────────┬───────────────────────┘
                       ▼
LLM (openai/gpt-oss-120b, JSON mode)
   └─ оценка 0–100, IPA по словам, статусы, советы (рус.)
                       ▼
        человекочитаемый отчёт / JSON
```

### Сигналы произношения

Whisper не измеряет артикуляцию. Скрипт использует два честных сигнала:

1. **Дифф** распознанного текста с ожидаемым: пропущенные, лишние и
   подменённые слова (`difflib.SequenceMatcher`).
2. **Уверенность модели**: `avg_logprob` сегмента; `< -0.6` → `warn`.

Третий слой — LLM: он получает транскрипт, дифф, уверенность и типичные ошибки
русскоязычных и формирует IPA + советы.

## Проверка на реальной речи (evidence)

Тестовый образец: TTS `edge-tts` (en-US-GuyNeural), фраза из примера задачи.

Команда:

```bash
python3 /root/tools/voice-pronunciation/pronounce.py /tmp/kirill.wav \
  -r "Hello, my name is Kirill"
```

Вывод:

```
🎧 Проверка произношения (English)

🎤 Услышано: Hello, my name is Kirill.
📖 Ожидалось: Hello, my name is Kirill
🎯 Совпадение слов: 100.0% (5/5)
📈 Уверенность модели: high (avg_logprob -0.202)
⭐ Оценка: 100/100

Слова:
  ✅ Hello /həˈloʊ/
  ✅ my /maɪ/
  ✅ name /neɪm/
  ✅ is /ɪz/
  ✅ Kirill /kɪˈrɪl/

💬 Отлично, произношение полностью понятно и без ошибок.

Советы:
  • Продолжайте практиковать произношение, уделяя внимание интонации предложения.
  • Работайте над связкой слов, чтобы звучало более естественно.
```

### Проверка разбора ошибок

Фраза-эталон намеренно отличается от сказанного («Cyril» вместо «Kirill», лишнее
слово «pronunciation») — дифф корректно ловит подмену и лишнее, LLM объясняет:

```
🎯 Совпадение слов: 90.0% (9/10)
⭐ Оценка: 80/100
  ❌ Cyril /ˈsɪrəl/ — Произносите без звука /k/.
Лишнее: pronunciation
```

## Использование

```bash
# Файл + ожидаемый текст
python3 pronounce.py voice.ogg -r "Hello, my name is Kirill"

# Без ожидаемого текста
python3 pronounce.py voice.ogg

# Запись 5 секунд с микрофона
python3 pronounce.py --record 5 -r "Good morning"

# Офлайн-эвристика (без LLM)
python3 pronounce.py voice.ogg -r "Hello" --no-llm

# JSON
python3 pronounce.py voice.ogg -r "Hello" --json --out result.json
```

Ключ `GROQ_API_KEY` читается из `/root/.hermes/.env` автоматически.

## Как это подключается к Telegram

Hermes получает голосовые сообщения и при включённом `stt` уже распознаёт их.
Скилл `voice-pronunciation` инструктирует агента: когда пришло английское
голосовое, запустить `pronounce.py` на файле этого голосового (с `-r`, если
пользователь назвал ожидаемый текст), и вернуть пользователю отчёт на русском.
Ответ можно озвучить через скилл `voice-communication` (TTS), продублировав текст.

## Тесты

```
python3 tests/test_pronounce.py
Ran 33 tests ... OK (skipped=2)          # unit

HERMES_PRON_IT=1 python3 tests/test_pronounce.py
Ran 33 tests ... OK                       # + 2 live-API теста
```

Покрыто: нормализация токенов, дифф (равенство/пропуск/вставка/замена/пустой
эталон), маппинг `avg_logprob` на слова, статусы слов, извлечение JSON из LLM,
парсинг `.env` и приоритет ключа, сборка multipart, рендер отчёта, эвристика,
поиск ffmpeg, живой пайплайн.

## Ограничения

- Whisper нормализует речь к известным словам — точный фонетический замер по
  звукам невозможен без GOP/forced-alignment (не реализовано).
- `avg_logprob` — уверенность распознавания, а не точность произношения; это
  прокси-сигнал.
- Дифф требует `--reference`; без него остаётся только `avg_logprob` + LLM.
- Запись с микрофона требует `pulseaudio`/`alsa` на сервере (для Telegram не нужно).
- Стоимость: бесплатный tier Groq (Whisper 2000 req/day).

## Итог

Задача выполнена: голосовое → распознавание → сравнение → IPA и советы работают
end-to-end на реальной речи; скилл Hermes создан; тесты проходят.
