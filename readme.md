# AI Chatbot

本專案為一個使用 FastAPI 串接大型語言模型（LLM）的後端服務，支援多種 LLM 提供者，包含 OpenAI 與 Gemini，透過不同的 Git 分支進行切換。

---

## 🔧 分支說明

- `feature/openai`：使用 OpenAI LLM。
- `feature/gemini`：使用 Gemini LLM。

切換分支後需重新建置 Docker 容器。

```bash
git checkout feature/openai  # 或 feature/gemini
docker-compose build
```

---

## 🚀 啟動專案

使用 Docker Compose 快速啟動：

```bash
docker-compose up -d
```

啟動後，FastAPI 服務將於預設的 `http://localhost:8000` 運行。

---

## 📂 專案結構（範例）

```
.
├── app/
│   ├── main.py
│   ├── routers/
│   └── services/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 📌 注意事項

- 請根據使用的 LLM 在 `.env` 檔案中設定對應的 API 金鑰與參數。
- 若切換分支未重新 build 容器，可能會造成依賴錯誤或無法正確串接。
