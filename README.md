# HomeAICore — AI Agent (`ai.agent`)

Первый **автономный прототип** Docker-модуля AI для Home-AI-Core (Module SDK v2).
Один модуль поддерживает **Ollama**, **OpenAI** и **OpenAI-compatible** серверы, включая llama.cpp и LM Studio (при наличии совместимого API).

## Что реализовано в 0.1.0

- Локальный веб-чат на русском языке, диалоги и история в SQLite `/data/ai.sqlite3`.
- Настраиваемые AI-провайдеры, переключение моделей и защищённый вход по Bearer-token.
- Чтение OpenAI Chat Completions и Ollama `/api/chat`; инструменты (tool calls) для моделей, которые их поддерживают.
- Предложения операций `modules.enable`, `modules.disable`, `modules.restart` с валидацией и очередью подтверждений.
- Сервис `/health`, независимый контейнер без root, Docker socket и host ports в установленном Core-модуле.
- Dockerfile, standalone Compose, CI-тесты и шаблон Manifest v2.

**Важно:** согласие на операцию переводит заявку в `approved_not_executed`. Модуль **не вызывает Core**, не запускает shell/Docker-команды и пока **не управляет устройствами**. Это намеренный безопасный первый этап, пока не будет защищённого Core → Module API с идентификацией пользователя и делегированными правами.

## Запуск для разработки

```bash
python -m unittest discover -s tests -v
mkdir -p /tmp/homeai-ai-data
AI_DATA_DIR=/tmp/homeai-ai-data AI_HOST=127.0.0.1 AI_PORT=8080 python -m ai_module.app
```

При первом запуске появится приватный файл `/tmp/homeai-ai-data/access-token` (режим `0600`). Откройте `http://127.0.0.1:8080/` и введите его содержимое.

Через Docker:

```bash
docker compose up --build -d
docker compose exec ai cat /data/access-token
```

Интерфейс доступен только на `http://127.0.0.1:8091/` локального Docker-хоста. Для удалённого доступа нужен аутентифицированный HTTPS gateway; **не публикуйте порт напрямую в интернет**.

## Подключение моделей

- **Ollama:** тип `ollama`, адрес сервера, например `http://<LAN_IP>:11434`, имя установленной модели. При запуске AI внутри Docker `127.0.0.1` — адрес **его контейнера**, а не хоста. Настройте доступ между контейнерами безопасно и ограничьте Ollama сетевыми правилами.
- **llama.cpp / LM Studio / иной локальный движок:** тип `openai_compatible`, адрес с `/v1`, модель.
- **OpenAI:** тип `openai`, **строго** `https://api.openai.com/v1`, модель и API-ключ. Для других облачных провайдеров используйте `openai_compatible` с HTTPS.

Ключи сохраняются локально в приватной базе в `/data` и не возвращаются API списка провайдеров. **Шифрование ключей на диске пока не добавлено**: обеспечьте права на volume и резервные копии. Не передавайте ключи на HTTP endpoint вне доверенной локальной сети.

## Публикация для Core

1. Сначала убедитесь, что CI прошёл. Отдельный workflow `Publish immutable AI image` собирает `linux/amd64` и `linux/arm64` и публикует `ghcr.io/deadsoulf/home-ai-core-ai` при теге `v*` или ручном запуске.
2. **Проверьте доступность GHCR package**: новый пакет по умолчанию может быть приватным. Для установки из Core образ должен быть доступен его Docker daemon без приватных credentials, иначе возникнет `unauthorized`.
3. Получите настоящий `sha256:...` OCI **manifest list / image index** после публикации, затем выполните:

```bash
python scripts/render_manifest.py sha256:<REAL_64_HEX_DIGEST> > manifest.json
```

4. Проверьте Manifest v2 и установите его через Core `/api/v1/modules/install` или веб-интерфейс; публикуйте в `home-ai-core_modules` только после проверки образа и совместимости.

`manifest.template.json` **не является установочным манифестом** до замены `IMAGE_DIGEST` реальным digest.

### Известные ограничения ядра

- Core Module SDK v2 запускает контейнер в сети `home-ai-modules`, **без host ports**, устройств и Docker socket, с read-only root и записью только в `/data`.
- `ui.navigation` создаёт пункт в левом меню, **но фактический защищённый reverse proxy для приложения модуля в Core ещё не реализован**. Сейчас пункт открывает заглушку. Интеграция API/UI требует отдельной доработки Core (не обходите RBAC/CSRF).
- Нет инъекции секретов/настроек в контейнер через Manifest v2. Сам AI имеет bootstrap-token в `/data`, но пока отсутствует подтверждённый защищённый способ передать его через UI ядра.
- Подключение внешних автономных агентов по MCP/A2A, инструменты камер/файлов/устройств, потоковая генерация, фоновые задачи и защищённый вызов Core API — следующие этапы; модели по совместимому API работают уже сейчас.

## Безопасность

- Нет автоматического выполнения предложенных AI операций.
- UI использует `textContent`, CSP, токен заголовка `Authorization`, без cookies/CORS; токен хранится только в памяти страницы.
- Пароли провайдеров не возвращаются из API; запросы ограничены размером и временем.
- Health открыт для Docker healthcheck и не содержит конфиденциальных данных.
- В standalone Compose публикуется только loopback-порт.

Контракт Core: [`docs/sdk/module-sdk-v2.md`](https://github.com/DeadSoulf/home-ai-core/blob/main/docs/sdk/module-sdk-v2.md).
