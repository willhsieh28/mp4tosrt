# mp4tosrt

[![Python Smoke Test](https://github.com/willhsieh28/mp4tosrt/actions/workflows/python-smoke-test.yml/badge.svg)](https://github.com/willhsieh28/mp4tosrt/actions/workflows/python-smoke-test.yml)
[![GitHub repo size](https://img.shields.io/github/repo-size/willhsieh28/mp4tosrt)](https://github.com/willhsieh28/mp4tosrt)
[![GitHub last commit](https://img.shields.io/github/last-commit/willhsieh28/mp4tosrt)](https://github.com/willhsieh28/mp4tosrt/commits/main)

`mp4tosrt` 是一個以 Gradio 建立的字幕產生工具，提供簡單的 Web 操作介面，能把本機 MP4 或 YouTube 影片快速轉成 `.srt` 字幕檔，並支援繁體中文優化、GPU 加速與外語字幕翻譯。

## 專案亮點

- 支援本機 `MP4` 上傳與 YouTube 網址下載
- 使用 `faster-whisper` 進行語音辨識
- 可依設備切換 `CPU` 或 `CUDA`
- 自動偵測語言並輸出對應字幕檔
- 中文字幕自動轉成台灣繁體中文用語
- 非中文字幕可翻譯為繁體中文
- 支援 Google 翻譯與本地 Ollama 翻譯模型
- 提供 Gradio 圖形介面，適合一般使用者直接操作

## 適合的使用情境

- 將課程錄影快速轉成字幕
- 整理訪談、會議或演講影片文字
- 把英文、日文、韓文等外語影片轉成繁體中文字幕
- 先自動產生字幕，再進一步手動校正

## Demo 截圖

你可以把專案畫面截圖放到 `docs/screenshots/`，再取消下面註解即可顯示在 GitHub 首頁。

```md
![mp4tosrt demo](docs/screenshots/demo-main.png)
```

建議至少準備這 2 張：

- `demo-main.png`：主畫面與參數設定區
- `demo-result.png`：字幕輸出與下載結果

## 功能流程

1. 選擇影片來源
2. 上傳本機 MP4，或貼上 YouTube 網址
3. 選擇 Whisper 模型大小
4. 選擇運算設備 `cpu` 或 `cuda`
5. 視需要選擇字幕翻譯引擎
6. 開始轉換並下載 `.srt` 字幕檔

## 環境需求

- Python 3.10 以上
- 建議使用虛擬環境
- 若要使用 GPU，請先安裝與本機 CUDA 相容的 PyTorch
- 若要使用 YouTube 下載功能，需安裝 `yt-dlp`
- 若要使用 Ollama 翻譯，需先在本機啟動 Ollama

## 安裝

### 1. 建立虛擬環境

```bash
python -m venv .venv
```

### 2. 啟用虛擬環境

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. 安裝套件

```bash
pip install -r requirements.txt
```

若你要使用 CUDA，建議先依照 PyTorch 官方安裝頁面安裝對應版本，再執行上面的需求安裝。

## 啟動方式

```bash
python mp4tosrt.py
```

啟動後請在瀏覽器開啟本機 Gradio 介面。

## 輸出說明

- 字幕檔預設輸出到 `SRT_Outputs/`
- 原始語言字幕會保留
- 若影片不是中文，會依設定額外產生繁體中文字幕檔
- 字幕檔會依偵測語言附上語系後綴，例如 `.en.srt`、`.ja.srt`、`.zh-TW.srt`

## 專案結構

```text
.
|-- .github/
|   |-- ISSUE_TEMPLATE/
|   |-- workflows/
|   `-- pull_request_template.md
|-- docs/
|   `-- screenshots/
|-- CHANGELOG.md
|-- CONTRIBUTING.md
|-- README.md
|-- requirements.txt
`-- mp4tosrt.py
```

## 已知限制

- `large` 或 `large-v3` 模型會使用較多記憶體
- YouTube 下載成功率會受 `yt-dlp` 與影片可用性影響
- Google 翻譯與 Ollama 翻譯品質會依原文內容與模型能力而不同
- 自動字幕仍可能需要人工校正專有名詞與標點

## Roadmap

- 支援批次處理多個影片
- 提供 CLI 模式
- 支援更多翻譯引擎
- 增加更完整的錯字修正與字幕切分策略
- 補齊測試與模組化結構

## 貢獻

如果你想幫忙改進這個專案，歡迎先閱讀 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 授權

目前尚未加入授權檔案。若你準備公開讓其他人使用與修改，建議補上 `MIT` 或 `Apache-2.0` 授權。
