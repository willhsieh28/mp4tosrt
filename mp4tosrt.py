import gradio as gr
import gc
from faster_whisper import WhisperModel
import datetime
import torch
import os
import tempfile
import pysrt
import requests
import re
import time
from deep_translator import GoogleTranslator
import sys

# 💡 新增：檢查是否已經安裝純 Python 版的 opencc
try:
    # 不論是 C++ 版還是純 Python 版，引入的名稱都是 opencc
    import opencc
except ImportError:
    print("❌ 錯誤：找不到 opencc 套件！")
    print("👉 請在終端機輸入以下指令來安裝純 Python 版本：")
    print("    pip install opencc-python-reimplemented")
    sys.exit(1) # 終止程式執行

# 加入這行防止函式庫衝突造成的閃退
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 建立一個全域保險箱，防止模型被系統回收而導致閃退
GLOBAL_MODEL_CACHE = {}

# --- 系統硬體偵測 ---
# 判斷電腦是否有支援 CUDA 的顯示卡，有的話就用顯示卡加速
HAS_GPU = torch.cuda.is_available()
DEVICE_CHOICES = ["cuda", "cpu"] if HAS_GPU else ["cpu"]
DEFAULT_DEVICE = "cuda" if HAS_GPU else "cpu"

# --- 偵測本地 Ollama 可用模型 ---
def get_ollama_models():
    """嘗試連線本地 Ollama API，取得可用模型清單"""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return models if models else []
    except Exception:
        pass
    return []

OLLAMA_MODELS = get_ollama_models()
HAS_OLLAMA = len(OLLAMA_MODELS) > 0

# 將秒數轉換為 SRT 字幕需要的時間格式 (時:分:秒,毫秒)
def format_timestamp(seconds):
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    milliseconds = int(td.microseconds / 1000)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# 呼叫本地 Ollama 模型進行翻譯
def translate_with_ollama(text, model_name, source_lang="auto", target_lang="繁體中文"):
    prompt = (
        f"你是一位專業翻譯員。請將以下{source_lang}文字翻譯成{target_lang}，"
        f"只輸出翻譯結果，不要加任何解釋或前綴：\n\n{text}"
    )
    try:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        if resp.status_code == 200:
            result = resp.json().get("response", "").strip()
            return result if result else text
    except Exception as e:
        print(f"[Ollama 翻譯錯誤] {e}")
    return text

# 下載 YouTube 影片
def download_youtube_video(url, log_fn=None):
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("未安裝 yt-dlp，請先執行: pip install yt-dlp")
    
    tmp_dir = tempfile.mkdtemp(prefix="mp4tosrt_")
    output_template = os.path.join(tmp_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
        'retries': 5,
    }
    
    if log_fn:
        log_fn("📥 正在從 YouTube 下載影片...")
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get('title', 'video')
        if info.get('requested_downloads'):
            downloaded_path = info['requested_downloads'][0]['filepath']
        else:
            for f in os.listdir(tmp_dir):
                if f.endswith('.mp4'):
                    downloaded_path = os.path.join(tmp_dir, f)
                    break
            else:
                raise RuntimeError("下載完成但找不到 MP4 檔案")
    
    if log_fn:
        log_fn(f"✅ YouTube 影片已下載: {title}")
    
    return downloaded_path

# 檢查輸入的網址是否為 YouTube 格式
def is_youtube_url(text):
    if not text:
        return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?(www\.)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/',
    ]
    return any(re.search(p, text) for p in patterns)

# --- 核心處理函式 ---
def process_video_to_srt(video_path, youtube_url, input_mode, model_size, device_type, 
                         translate_engine, ollama_model, progress=gr.Progress()):
    
    if input_mode == "YouTube 網址":
        if not youtube_url or not youtube_url.strip():
            yield None, "⚠️ 請輸入 YouTube 網址！"
            return
        if not is_youtube_url(youtube_url.strip()):
            yield None, "⚠️ 請輸入有效的 YouTube 網址！"
            return
    else:
        if video_path is None:
            yield None, "⚠️ 請先上傳 MP4 影片檔案！"
            return
    
    def log(msg, p_val=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
        if p_val is not None:
            progress(p_val, desc=msg)

    try:
        log("⏳ 準備開始作業...", 0.05)
        
        if input_mode == "YouTube 網址":
            log("📥 開始下載 YouTube 影片...", 0.1)
            try:
                video_path = download_youtube_video(youtube_url.strip(), log_fn=lambda m: log(m))
            except Exception as e:
                yield None, f"❌ YouTube 影片下載失敗：{str(e)}"
                return
            log("✅ YouTube 影片下載完成！", 0.2)
        
        # 💡 初始化 opencc 轉換器
        # 's2twp' 代表簡體轉為台灣繁體，並且會一併轉換兩岸慣用語
        converter = opencc.OpenCC('s2twp')
        
        compute_type = "float16" if device_type == "cuda" else "int8"
        log(f"🧠 正在載入 {model_size} 模型至 {device_type.upper()}...", 0.3)
        
        # 將模型鎖在保險箱中，避免重複載入吃光記憶體
        global GLOBAL_MODEL_CACHE
        cache_key = f"{model_size}_{device_type}"
        
        if cache_key not in GLOBAL_MODEL_CACHE:
            GLOBAL_MODEL_CACHE[cache_key] = WhisperModel(model_size, device=device_type, compute_type=compute_type)
        model = GLOBAL_MODEL_CACHE[cache_key]
        
        log("🎧 語音辨識中 (使用 faster-whisper + VAD 精準切割)...", 0.5)
        prompt_text = "這是一段繁體中文字幕，請使用繁體中文輸出。專有名詞參考：Python, Gradio, OpenAI, 機器學習, 人工智慧。"
        
        # 進行語音辨識
        log("⏳ 開始語音辨識，請保持頁面開啟...")
        segments_gen, info = model.transcribe(
            video_path, 
            initial_prompt=prompt_text,
            beam_size=5,
            condition_on_previous_text=True, 
            vad_filter=False, 
            no_speech_threshold=0.9, 
            word_timestamps=True 
        )
        
        segments_list = []
        log("⏳ 語音辨識進行中，請保持頁面開啟...", 0.55)
        
        for segment in segments_gen:
            segments_list.append(segment)
            log(f"🗣️ 辨識進度：已處理至 {format_timestamp(segment.end)}", 0.6)
            time.sleep(0.01)
            
        detected_lang = info.language
        log(f"🔍 第一輪辨識結果：語言={detected_lang}，段落數={len(segments_list)}")

        if len(segments_list) == 0:
            log("⚠️ VAD 過濾後無段落，關閉 VAD 重新辨識...", 0.6)
            segments_gen2, info2 = model.transcribe(
                video_path,
                initial_prompt=None,
                beam_size=5,
                condition_on_previous_text=False,
                vad_filter=False,
            )
            segments_list = []
            for segment in segments_gen2:
                segments_list.append(segment)
                log(f"🗣️ 重新辨識進度：已處理至 {format_timestamp(segment.end)}", 0.65)
                time.sleep(0.01)

            detected_lang = info2.language
            log(f"🔍 第二輪辨識結果：語言={detected_lang}，段落數={len(segments_list)}")
        
        lang_map = {
            "zh": ("繁體中文", "zh-TW"), "en": ("英文", "en"),
            "ja": ("日文", "ja"), "ko": ("韓文", "ko")
        }
        lang_name, lang_suffix = lang_map.get(detected_lang, (f"未知語言({detected_lang})", detected_lang))
        
        log(f"📝 辨識語言為 {lang_name}，共 {len(segments_list)} 段，產生 SRT 字幕檔...", 0.85)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        for ext in [".zh", ".zh-TW", ".en", ".ja", ".ko"]:
            if base_name.endswith(ext):
                base_name = base_name[:-len(ext)]
                break
                
        # 建立安全的輸出資料夾
        out_dir = os.path.join(os.getcwd(), "SRT_Outputs")
        os.makedirs(out_dir, exist_ok=True)
        output_filename = os.path.join(out_dir, f"{base_name}.{lang_suffix}.srt")
        
        # 自訂錯字修正字典
        custom_corrections = {
            "因該": "應該", "的化": "的話", "做品": "作品", "視頻": "影片",
        }
        
        with open(output_filename, "w", encoding="utf-8") as srt_file:
            srt_index = 1
            for segment in segments_list:
                if getattr(segment, "words", None):
                    chunk_start = segment.words[0].start
                    chunk_text = ""
                    
                    for word_obj in segment.words:
                        chunk_text += word_obj.word
                        word_text = word_obj.word
                        
                        has_end_punct = any(p in word_text for p in ['.', '!', '?', '。', '！', '？'])
                        has_comma = any(p in word_text for p in [',', '，'])
                        
                        if has_end_punct or (has_comma and len(chunk_text) > 25) or len(chunk_text) > 50:
                            start_time = format_timestamp(chunk_start)
                            end_time = format_timestamp(word_obj.end)
                            
                            out_text = chunk_text.strip()
                            if detected_lang == "zh":
                                # 💡 使用純 Python 版 opencc 進行繁簡轉換
                                out_text = converter.convert(out_text)
                                for wrong_word, correct_word in custom_corrections.items():
                                    out_text = out_text.replace(wrong_word, correct_word)
                                    
                            srt_file.write(f"{srt_index}\n{start_time} --> {end_time}\n{out_text}\n\n")
                            srt_index += 1
                            
                            chunk_text = ""
                            chunk_start = word_obj.end
                            
                    if chunk_text.strip():
                        start_time = format_timestamp(chunk_start)
                        end_time = format_timestamp(segment.end)
                        out_text = chunk_text.strip()
                        if detected_lang == "zh":
                            # 💡 再次套用 opencc 轉換剩餘碎片
                            out_text = converter.convert(out_text)
                            for wrong_word, correct_word in custom_corrections.items():
                                out_text = out_text.replace(wrong_word, correct_word)
                        srt_file.write(f"{srt_index}\n{start_time} --> {end_time}\n{out_text}\n\n")
                        srt_index += 1
                        
                else:
                    srt_file.write(f"{srt_index}\n")
                    start_time = format_timestamp(segment.start)
                    end_time = format_timestamp(segment.end)
                    srt_file.write(f"{start_time} --> {end_time}\n")
                    
                    text = segment.text.strip()
                    if detected_lang == "zh":
                        # 💡 如果沒有啟用單字時間軸，這裡也會套用 opencc
                        text = converter.convert(text)
                        for wrong_word, correct_word in custom_corrections.items():
                            text = text.replace(wrong_word, correct_word)
                    srt_file.write(f"{text}\n\n")
                    srt_index += 1
                    
        output_files = [output_filename]
        
        # 處理外文翻譯邏輯
        if detected_lang != "zh":
            if translate_engine == "Ollama (本地模型)":
                selected_model = ollama_model if ollama_model else (OLLAMA_MODELS[0] if OLLAMA_MODELS else "")
                if not selected_model:
                    yield None, "❌ 未選擇 Ollama 模型，且本地無可用模型。"
                    return
                
                log(f"🤖 使用 Ollama 模型 [{selected_model}] 翻譯中，請稍候...", 0.9)
                subs = pysrt.open(output_filename, encoding='utf-8')
                total_subs = len(subs)
                
                for idx, sub in enumerate(subs):
                    if sub.text.strip():
                        t_text = translate_with_ollama(sub.text, selected_model, source_lang=lang_name)
                        # 💡 翻譯後再次確保用語轉換
                        t_text = converter.convert(t_text)
                        for wrong_word, correct_word in custom_corrections.items():
                            t_text = t_text.replace(wrong_word, correct_word)
                        sub.text = t_text
                    
                    if total_subs > 0:
                        p = 0.9 + (idx / total_subs) * 0.09
                        if idx % max(1, total_subs // 20) == 0:
                            log(f"🤖 Ollama 翻譯進度：{idx+1}/{total_subs}", p)
                            time.sleep(0.01)
                
                translated_filename = os.path.join(out_dir, f"{base_name}.zh-TW.srt")
                subs.save(translated_filename, encoding='utf-8')
                output_files.append(translated_filename)
                status_msg = f"✅ 轉換成功！已透過 Ollama [{selected_model}] 產生繁體中文字幕檔。"
                
            else:
                log("🌍 偵測到非中文，使用 Google 翻譯進行繁體中文翻譯中，請稍候...", 0.92)
                subs = pysrt.open(output_filename, encoding='utf-8')
                translator = GoogleTranslator(source='auto', target='zh-TW')
                
                total_subs = len(subs)
                for idx, sub in enumerate(subs):
                    if sub.text.strip():
                        t_text = translator.translate(sub.text)
                        # 💡 翻譯後再次確保用語轉換
                        t_text = converter.convert(t_text)
                        for wrong_word, correct_word in custom_corrections.items():
                            t_text = t_text.replace(wrong_word, correct_word)
                        sub.text = t_text
                    
                    if idx % 5 == 0:
                        progress_val = 0.92 + (idx / total_subs) * 0.07
                        log(f"🌍 Google 翻譯進度：{idx+1}/{total_subs}", progress_val)
                        time.sleep(0.01)
                        
                translated_filename = os.path.join(out_dir, f"{base_name}.zh-TW.srt")
                subs.save(translated_filename, encoding='utf-8')
                output_files.append(translated_filename)
                status_msg = f"✅ 轉換成功！（使用 {model_size} 模型）並已透過 Google 翻譯產生繁體中文字幕檔。"
        else:
            status_msg = f"✅ 轉換成功！（使用 {model_size} 模型 + {device_type.upper()}）。"
            
        log(status_msg, 1.0)
        
        output_files = [os.path.abspath(f) for f in output_files]
        time.sleep(0.5) 
        
        yield output_files, status_msg
        
    except Exception as e:
        error_msg = f"❌ 轉換過程中發生錯誤：{str(e)}"
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {error_msg}")
        yield None, error_msg

# --- 建立 Gradio 網頁介面 ---
with gr.Blocks(title="MP4 轉 SRT 產生器") as demo:
    gr.Markdown("# 🎬 MP4 轉 SRT 字幕產生器 (支援 GPU 加速 & 台灣繁體中文)")
    
    status_parts = []
    if HAS_GPU:
        status_parts.append("🟢 **GPU：已偵測到，可使用硬體加速**")
    else:
        status_parts.append("🔴 **GPU：未偵測到，將使用純 CPU 運算**")
    if HAS_OLLAMA:
        status_parts.append(f"🟢 **Ollama：已連線，可用模型 {len(OLLAMA_MODELS)} 個**")
    else:
        status_parts.append("🟡 **Ollama：未偵測到（僅支援 Google 翻譯）**")
    
    gr.Markdown(" | ".join(status_parts))
    input_mode_state = gr.State("上傳檔案")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 影片來源")
            with gr.Tabs() as input_tabs:
                with gr.Tab("📁 上傳檔案") as tab_upload:
                    video_input = gr.File(label="上傳 MP4 影片", file_types=[".mp4"], type="filepath")
                with gr.Tab("🔗 YouTube 網址") as tab_youtube:
                    youtube_url_input = gr.Textbox(
                        label="YouTube 網址",
                        placeholder="https://www.youtube.com/watch?v=...",
                        lines=1
                    )
        
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=["tiny", "base", "small", "medium", "large", "large-v3"], 
                value="large-v3", 
                label="2. 選擇辨識模型",
                info="推薦使用 large-v3 以獲得最高辨識準確率"
            )
            device_dropdown = gr.Dropdown(
                choices=DEVICE_CHOICES,
                value=DEFAULT_DEVICE,
                label="3. 選擇運算設備",
                info="cuda = 顯示卡加速；cpu = 傳統處理器運算"
            )
            
            translate_engine_choices = ["Google 翻譯"]
            if HAS_OLLAMA:
                translate_engine_choices.append("Ollama (本地模型)")
            
            translate_engine_dropdown = gr.Dropdown(
                choices=translate_engine_choices,
                value="Google 翻譯",
                label="4. 字幕翻譯引擎（非中文影片適用）",
                info="選擇將外語字幕翻譯成繁體中文的引擎"
            )
            
            ollama_model_dropdown = gr.Dropdown(
                choices=OLLAMA_MODELS if OLLAMA_MODELS else ["(未偵測到 Ollama)"],
                value=OLLAMA_MODELS[0] if OLLAMA_MODELS else "(未偵測到 Ollama)",
                label="4-1. 選擇 Ollama 模型",
                info="僅在翻譯引擎選擇「Ollama (本地模型)」時生效",
                interactive=HAS_OLLAMA
            )
    
    with gr.Row():
        start_btn = gr.Button("▶️ 5. 開始轉換", variant="primary")
        reset_btn = gr.ClearButton(value="🔄 6. 重新開始")
        
    with gr.Row():
        status_text = gr.Textbox(label="系統狀態", interactive=False)
        download_file = gr.File(label="📥 7. 下載 SRT 字幕檔", file_count="multiple")

    # --- 互動邏輯 ---
    tab_upload.select(fn=lambda: "上傳檔案", inputs=[], outputs=[input_mode_state])
    tab_youtube.select(fn=lambda: "YouTube 網址", inputs=[], outputs=[input_mode_state])
    
    start_btn.click(
        fn=process_video_to_srt,
        inputs=[video_input, youtube_url_input, input_mode_state, 
                model_dropdown, device_dropdown, translate_engine_dropdown, ollama_model_dropdown],
        outputs=[download_file, status_text]
    )
    reset_btn.add([video_input, youtube_url_input, status_text])

if __name__ == "__main__":
    demo.queue()
    # 關閉 share 避免本地端口衝突，確保穩定執行
    demo.launch(server_name="127.0.0.1", share=False)