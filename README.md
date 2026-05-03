# mp4tosrt

`mp4tosrt` 是一個以 Gradio 建立的字幕產生工具，可將本機 MP4 影片或 YouTube 影片轉為 `.srt` 字幕檔，並支援 GPU 加速、繁體中文優化，以及外語字幕翻譯成繁體中文。

## 功能特色

- 支援上傳本機 `MP4` 影片
- 支援輸入 YouTube 網址後自動下載影片
- 使用 `faster-whisper` 進行語音辨識
- 自動偵測語言並輸出對應語系字幕檔
- 中文字幕會轉為台灣繁體中文用語
- 非中文字幕可使用 Google 翻譯或本地 Ollama 模型翻譯為繁體中文
- 提供 Gradio Web UI，操作直覺

## 執行畫面流程

1. 選擇影片來源
2. 選擇 Whisper 模型大小
3. 選擇運算設備 `cpu` 或 `cuda`
4. 視需要選擇翻譯引擎
5. 產生並下載 `.srt` 字幕檔

## 環境需求

- Python 3.10 以上
- 建議使用虛擬環境
- 若要使用 GPU，請先安裝與本機 CUDA 相容的 PyTorch
- 若要使用 YouTube 下載功能，需安裝 `yt-dlp`
- 若要使用 Ollama 翻譯，需先在本機啟動 Ollama

## 安裝方式

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

若你需要 CUDA 版本的 PyTorch，請依照官方說明安裝對應版本後，再補裝其餘套件。

## 啟動方式

```bash
python mp4tosrt.py
```

啟動後請在瀏覽器開啟本機 Gradio 介面。

## 輸出結果

- 字幕檔預設輸出到 `SRT_Outputs/`
- 原始語言字幕會保留
- 若啟用翻譯，會額外產生繁體中文字幕檔

## 專案結構

```text
.
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |-- workflows/
|   `-- pull_request_template.md
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- README.md
|-- requirements.txt
`-- mp4tosrt.py
```

## 已知限制

- `large` 或 `large-v3` 模型會佔用較多記憶體
- YouTube 下載成功率取決於 `yt-dlp` 與影片可用性
- Google 翻譯與 Ollama 翻譯品質會依來源內容與模型而不同

## 後續可擴充方向

- 批次處理多個影片
- 自訂輸出檔名格式
- 提供 CLI 模式
- 支援更多翻譯引擎

