from __future__ import annotations
import os, re, json, subprocess, threading, shutil, time
from pathlib import Path
from typing import List, Dict, Any # Added Any for broader type hinting
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------- 설정 --------
FFMPEG_BIN = "ffmpeg"
FFPROBE_BIN = "ffprobe"
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm"}
PBF_EXT = ".pbf"
SCRIPT_DIR = Path(__file__).resolve().parent
TOTAL_CORES = os.cpu_count() or 1
DEFAULT_CORES = max(1, TOTAL_CORES // 2)

# -------- 유틸 함수 --------
def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='ignore')

def is_tool_available(name: str) -> bool:
    """Checks whether `name` is on PATH and is executable."""
    return shutil.which(name) is not None

def ffprobe_get_chapters(video: Path) -> List[Dict]:
    cmd = [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_chapters", str(video)]
    try:
        res = run(cmd)
        data = json.loads(res.stdout or "{}")
        chaps = []
        for idx, ch in enumerate(data.get("chapters", []), start=1):
            start = float(ch.get("start_time", ch.get("start", 0)))
            title = ch.get("tags", {}).get("title", f"Chapter{idx:02d}")
            chaps.append({"start": start, "title": title})
        return sorted(chaps, key=lambda x: x["start"])
    except:
        return []

def ffprobe_get_audio_streams(video: Path) -> List[int]:
    """Returns a list of 0-based audio stream indices for FFmpeg mapping (e.g., 0, 1, 2...)."""
    cmd = [FFPROBE_BIN, "-v", "error", "-print_format", "json", "-show_streams", str(video)]
    try:
        res = run(cmd)
        data = json.loads(res.stdout or "{}")
        audio_stream_indices = []
        audio_counter = 0
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "audio":
                audio_stream_indices.append(audio_counter)
                audio_counter += 1
        return audio_stream_indices
    except Exception as e:
        print(f"Error getting audio streams: {e}")
        return []


def parse_pbf(pbf_path: Path) -> List[Dict]:
    markers = []
    if not pbf_path.exists(): return markers
    try:
        content = pbf_path.read_bytes().decode('utf-16-le', errors='ignore')
    except:
        return markers
    pat = re.compile(r"\d+=(\d+)\*([^\*]+)\*")
    # Corrected from .iter to .finditer
    for m in pat.finditer(content):
        ms, title = m.group(1), m.group(2).strip().replace('/', '_')
        try:
            start = int(ms) / 1000.0
            markers.append({"start": start, "title": title})
        except:
            continue
    return sorted(markers, key=lambda x: x['start'])

def seconds_to_tc(sec: float) -> str:
    h, m, s = int(sec//3600), int((sec%3600)//60), int(sec%60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def sanitize_filename(name: str) -> str:
    return re.sub(r"[\\/:*?<>|]", "_", name)

def get_video_duration(video_path: Path) -> float:
    cmd = [FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)]
    try:
        return float(run(cmd).stdout.strip())
    except:
        return 0.0

def analyze_single_video(video_path: Path) -> Dict[str, Any]:
    """단일 비디오 파일의 메타데이터를 분석하여 반환"""
    try:
        pbf = video_path.with_suffix(PBF_EXT)
        pbf_chaps_count = len(parse_pbf(pbf)) if pbf.exists() else 0
        ff_chaps_full = ffprobe_get_chapters(video_path)
        ff_chaps_count = len(ff_chaps_full)
        ff_audio_streams = ffprobe_get_audio_streams(video_path)
        audio_count = len(ff_audio_streams)
        duration = get_video_duration(video_path)
        
        return {
            "video": video_path,
            "pbf": pbf if pbf.exists() else None,
            "pbf_chaps_count": pbf_chaps_count,
            "ff_chapters_full": ff_chaps_full,
            "ff_chaps_count": ff_chaps_count,
            "ff_audio_streams": ff_audio_streams,
            "audio_count": audio_count,
            "duration": duration,
            "analyzed": True
        }
    except Exception as e:
        print(f"Error analyzing {video_path}: {e}")
        return {
            "video": video_path,
            "pbf": None,
            "pbf_chaps_count": 0,
            "ff_chapters_full": [],
            "ff_chaps_count": 0,
            "ff_audio_streams": [],
            "audio_count": 0,
            "duration": 0.0,
            "analyzed": False,
            "error": str(e)
        }

# -------- GUI 애플리케이션 --------
class VideoToolkitApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("영상 도구 툴킷")
        self.geometry("1000x680") # Changed width from 1200 to 1000
        self.resizable(False, False)

        self.files: Dict[str, Dict] = {}
        self.mode = tk.StringVar(value="trim")
        self.trim_strategy = tk.StringVar(value="pbf")
        # self.trim_audio = tk.BooleanVar(value=True) # REMOVED: "오디오 트랙 유지" 옵션 제거
        self.between_only = tk.BooleanVar(value=False)
        self.use_gpu = tk.BooleanVar(value=False)
        self.cpu_cores = tk.IntVar(value=DEFAULT_CORES)
        
        self.timer_id = None
        self.start_time = 0
        self.conversion_thread: threading.Thread | None = None
        self.mode_frame = None # Initialize mode_frame here
        self.audio_track_vars: Dict[int, tk.BooleanVar] = {} # Store BooleanVars for audio tracks
        self.audio_track_checkboxes: List[ttk.Checkbutton] = [] # Store checkbox widgets for easy clearing
        self.audio_track_label = None # To store the "오디오 트랙 선택:" label
        
        # 캐싱을 위한 변수들
        self.file_cache: Dict[Path, Dict] = {}  # 파일별 메타데이터 캐시
        self.scan_progress_var = tk.StringVar(value="준비됨")
        self.scan_thread: threading.Thread | None = None

        self._build()
        self.mode.trace_add('write', lambda *args: self._update_mode_ui())
        self.trim_strategy.trace_add('write', lambda *args: self._on_file_select()) # Added trace for trim_strategy
        self.between_only.trace_add('write', lambda *args: self._on_file_select())
        # self.trim_audio.trace_add('write', self._on_trim_audio_toggle) # REMOVED: "오디오 트랙 유지" 옵션 제거
        self._refresh_list()

    def _build(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="폴더 내 파일", font=("Segoe UI",12,"bold")).pack()
        
        # New frame for listbox and scrollbar
        list_frame = ttk.Frame(left)
        list_frame.pack(side="left", fill="y", expand=True)

        self.listbox = tk.Listbox(list_frame, width=80, height=30, font=("Consolas", 10)) # Increased width, set font
        self.listbox.pack(side="left", fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_file_select)
        sb = ttk.Scrollbar(list_frame, command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=sb.set)

        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(10,0))

        self.mode_frame = ttk.LabelFrame(right, text="작업 모드") # Assign to self.mode_frame
        self.mode_frame.pack(fill="x")
        ttk.Radiobutton(self.mode_frame, text="영상 자르기", variable=self.mode, value="trim").pack(anchor="w")
        ttk.Radiobutton(self.mode_frame, text="이름 바꾸기", variable=self.mode, value="rename").pack(anchor="w")

        self.trim_fr = ttk.LabelFrame(right, text="자르기 옵션")
        self.trim_fr.pack(fill="x", pady=(10,0))
        ttk.Radiobutton(self.trim_fr, text="PBF 기준만", variable=self.trim_strategy, value="pbf").pack(anchor="w")
        ttk.Radiobutton(self.trim_fr, text="챕터 기준만", variable=self.trim_strategy, value="chapter").pack(anchor="w")
        ttk.Radiobutton(self.trim_fr, text="교차(PBF+챕터)", variable=self.trim_strategy, value="both").pack(anchor="w")
        ttk.Checkbutton(self.trim_fr, text="챕터 사이만", variable=self.between_only).pack(anchor="w")
        
        # Audio options frame - now a separate LabelFrame
        self.audio_config_fr = ttk.LabelFrame(right, text="오디오 옵션")
        self.audio_config_fr.pack(fill="x", pady=(10,0))
        # ttk.Checkbutton(self.audio_config_fr, text="오디오 트랙 유지", variable=self.trim_audio).pack(anchor="w") # REMOVED
        
        # Container for dynamic audio track checkboxes
        self.audio_track_checkbox_container_fr = ttk.Frame(self.audio_config_fr)
        self.audio_track_checkbox_container_fr.pack(anchor="w", fill="x", pady=(5,0))
        
        # Create the "오디오 트랙 선택:" label once
        self.audio_track_label = ttk.Label(self.audio_track_checkbox_container_fr, text="오디오 트랙 선택:")
        self.audio_track_label.pack(side="left", anchor="w")


        self.cpu_fr = ttk.LabelFrame(right, text=f"CPU 코어 수 (총 {TOTAL_CORES}개)")
        self.cpu_fr.pack(fill="x", pady=(10,0))
        ttk.Spinbox(self.cpu_fr, from_=1, to=TOTAL_CORES, textvariable=self.cpu_cores, width=5).pack(anchor="w")
        ttk.Checkbutton(self.cpu_fr, text="GPU 가속 사용", variable=self.use_gpu).pack(anchor="w")

        self.progress = ttk.Progressbar(right, mode="determinate")
        self.progress.pack(fill="x", pady=(10,0))
        self.lbl_elapsed = ttk.Label(right, text="경과: 00:00:00")
        self.lbl_elapsed.pack(anchor="w", pady=(2,5))
        self.lbl_scan_progress = ttk.Label(right, textvariable=self.scan_progress_var, font=("Segoe UI", 9))
        self.lbl_scan_progress.pack(anchor="w", pady=(0,10))

        bf = ttk.Frame(right) # Changed to ttk.Frame for consistency
        bf.pack(fill="x", pady=(5, 15)) # Increased pady for more space below buttons

        self.btn_convert = ttk.Button(bf, text="변환 실행", command=self._on_convert) # Changed to ttk.Button
        self.btn_convert.pack(side="left", padx=5)

        self.btn_refresh = ttk.Button(bf, text="리스트 새로고침", command=self._refresh_list) # Changed to ttk.Button
        self.btn_refresh.pack(side="left", padx=5)

        nf = ttk.Notebook(right)
        nf.pack(fill="both", expand=True)
        tab1 = ttk.Frame(nf)
        nf.add(tab1, text="챕터 정보")
        self.chap_text = tk.Text(tab1, state="disabled", wrap="none")
        self.chap_text.pack(fill="both", expand=True)
        tab2 = ttk.Frame(nf)
        nf.add(tab2, text="최종 결과")
        self.res_text = tk.Text(tab2, state="disabled", wrap="none")
        self.res_text.pack(fill="both", expand=True)

        nf.pack(fill="both", expand=True)

    def _on_convert(self):
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("선택 필요", "먼저 파일을 선택하세요.")
            return
        # Ensure we don't try to process the header row
        if sel[0] == 0:
            messagebox.showwarning("선택 필요", "유효한 파일을 선택하세요.")
            return

        selected_label = self.listbox.get(sel[0])
        info = self.files[selected_label]

        if self.mode.get() == "rename":
            self._open_rename_dialog(info)
        else: # trim mode
            # Check for audio track selection
            selected_audio_tracks = [idx for idx, var in self.audio_track_vars.items() if var.get()]
            if not selected_audio_tracks and len(info['ff_audio_streams']) > 0: # Only warn if there are actual audio streams in the file
                messagebox.showwarning("오디오 트랙 선택 필요", "하나 이상의 오디오 트랙을 선택해야 합니다.")
                return

            # Re-evaluate chapters based on current settings for conversion validation
            strat = self.trim_strategy.get()
            chapters_raw: List[Dict] = []
            
            # Fetch actual chapter data from the file for display/processing
            full_chapters_from_file = ffprobe_get_chapters(info['video']) 
            if strat in ("chapter", "both"): chapters_raw.extend(full_chapters_from_file)
            if strat in ("pbf", "both") and info.get('pbf'): chapters_raw.extend(parse_pbf(info['pbf']))
            
            uniq = {c['start']: c['title'] for c in chapters_raw}
            sorted_chaps = sorted([{'start': s, 'title': t} for s, t in uniq.items()], key=lambda x: x['start'])

            # Determine trimmed chapters for validation
            trimmed_chaps = sorted_chaps[:-1] if self.between_only.get() and len(sorted_chaps) > 1 else sorted_chaps
            
            if not trimmed_chaps:
                messagebox.showwarning("변환 불가", "선택된 자르기 옵션에 해당하는 챕터/마커를 찾을 수 없습니다. 다른 옵션을 선택하거나 파일을 확인해주세요.")
                return

            # Start conversion in a new thread
            self._lock_ui_for_conversion()
            self._start_timer()
            
            self.res_text.config(state='normal')
            self.res_text.delete('1.0', tk.END)
            self.res_text.insert(tk.END, f"'{selected_label}' 변환 작업을 시작합니다...\n")
            self.res_text.config(state='disabled')

            # Reset progress bar
            self.progress['value'] = 0

            self.conversion_thread = threading.Thread(target=self._start_conversion, args=(info['video'], sorted_chaps))
            self.conversion_thread.start()


    def _start_conversion(self, video_path: Path, sorted_chapters: List[Dict]):
        try:
            # Create a subfolder named after the original video file (stem)
            output_subfolder = SCRIPT_DIR / video_path.stem
            output_subfolder.mkdir(parents=True, exist_ok=True)

            video_duration = get_video_duration(video_path)
            
            self.res_text.config(state='normal')
            self.res_text.insert(tk.END, f"비디오 길이: {seconds_to_tc(video_duration)}\n")
            self.res_text.insert(tk.END, "챕터 정보:\n")
            for i, chapter in enumerate(sorted_chapters):
                self.res_text.insert(tk.END, f"  {i+1}. {seconds_to_tc(chapter['start'])} - {chapter['title']}\n")

            trimmed_segments = []
            if self.between_only.get() and len(sorted_chapters) > 1:
                # If "챕터 사이만" is checked, we cut between existing chapters
                for i in range(len(sorted_chapters) - 1):
                    start = sorted_chapters[i]['start']
                    end = sorted_chapters[i+1]['start']
                    title = sorted_chapters[i]['title'] # Use the start chapter's title
                    trimmed_segments.append({'start': start, 'end': end, 'title': title})
            else:
                # Cut from each chapter to the next or end of video
                for i, chapter in enumerate(sorted_chapters):
                    start = chapter['start']
                    end = sorted_chapters[i+1]['start'] if i + 1 < len(sorted_chapters) else video_duration
                    title = chapter['title']
                    trimmed_segments.append({'start': start, 'end': end, 'title': title})

            if not trimmed_segments:
                 self.res_text.insert(tk.END, "생성할 비디오 세그먼트가 없습니다.\n")
                 return
            
            # Set progress bar maximum based on number of segments
            self.progress['maximum'] = len(trimmed_segments)


            self.res_text.insert(tk.END, f"\n변환할 세그먼트 (출력 폴더: {output_subfolder.name}):\n")
            for idx, segment in enumerate(trimmed_segments):
                # Ensure output filename is unique and safe, following 01_ChapterName.ext format
                base_filename = sanitize_filename(f"{idx+1:02d}_{segment['title']}")
                output_filename = f"{base_filename}{video_path.suffix}"
                output_path = output_subfolder / output_filename # Save inside the new subfolder

                # Handle potential duplicate filenames (e.g., if titles are identical)
                counter = 1
                while output_path.exists():
                    output_filename = f"{base_filename}_{counter}{video_path.suffix}"
                    output_path = output_subfolder / output_filename
                    counter += 1
                
                self.res_text.insert(tk.END, f"  - {output_path.name} (시작: {seconds_to_tc(segment['start'])}, 끝: {seconds_to_tc(segment['end'])})\n")
                
                cmd = [
                    FFMPEG_BIN,
                    "-ss", seconds_to_tc(segment['start']),
                    "-i", str(video_path),
                    "-to", seconds_to_tc(segment['end'] - segment['start']),
                    "-avoid_negative_ts", "make_zero",
                    "-fflags", "+genpts",
                ]
                
                # Video stream mapping
                cmd.extend(["-map", "0:v"])

                # Handle audio stream mapping based on selection
                selected_audio_tracks = [idx for idx, var in self.audio_track_vars.items() if var.get()]
                if selected_audio_tracks:
                    for track_idx in selected_audio_tracks:
                        cmd.extend(["-map", f"0:a:{track_idx}"]) # Use the 0-based index directly
                    cmd.extend(["-c:a", "copy"]) # Copy selected audio tracks
                else:
                    cmd.extend(["-an"]) # No audio track if nothing selected

                # Handle video codec
                if self.use_gpu.get():
                    cmd.extend(["-c:v", "h264_nvenc"]) # Or "hevc_nvenc" for H.265
                    self.res_text.insert(tk.END, "INFO: GPU 가속 (h264_nvenc) 사용.\n")
                else:
                    cmd.extend(["-c:v", "copy"]) # Explicitly copy video stream by default

                cmd.append(str(output_path))
                
                self.res_text.insert(tk.END, f"명령 실행: {' '.join(cmd)}\n")
                process = run(cmd)
                self.res_text.insert(tk.END, f"STDOUT:\n{process.stdout}\n")
                self.res_text.insert(tk.END, f"STDERR:\n{process.stderr}\n")
                if process.returncode != 0:
                    self.res_text.insert(tk.END, f"파일 '{output_path.name}' 변환 중 오류 발생!\n")
                    raise RuntimeError(f"FFmpeg 오류: {process.stderr}")
                else:
                    self.res_text.insert(tk.END, f"파일 '{output_path.name}' 변환 완료.\n")
                
                # Update progress bar for each completed segment
                self.progress['value'] = idx + 1
                self.update_idletasks() # Force UI update

            self.res_text.insert(tk.END, "\n모든 변환 작업이 완료되었습니다.\n")
            messagebox.showinfo("변환 완료", "모든 영상 변환 작업이 완료되었습니다.") # Conversion completion notification
        except Exception as e:
            self.res_text.insert(tk.END, f"\n오류 발생: {e}\n")
            messagebox.showerror("변환 오류", f"변환 중 오류가 발생했습니다: {e}")
        finally:
            self.res_text.config(state='disabled')
            self._stop_timer()
            self.progress['value'] = 0 # Reset progress bar
            self.conversion_thread = None # Ensure the thread is marked as finished
            # Schedule unlock on main thread to ensure UI updates are safe
            self.after(100, self._unlock_ui_after_conversion) 


    def _lock_ui_for_conversion(self):
        # Disable all relevant UI elements by targeting specific interactive widgets
        
        # Trim options
        for rb in self.trim_fr.winfo_children():
            if isinstance(rb, (ttk.Radiobutton, ttk.Checkbutton)): # Use tuple for multiple types
                rb.config(state='disabled')
        
        # CPU options
        for child in self.cpu_fr.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Checkbutton)): # Use tuple for multiple types
                child.config(state='disabled')

        # Audio options
        # self.audio_config_fr.winfo_children()[0].config(state='disabled') # REMOVED: "오디오 트랙 유지" 옵션 제거
        for cb in self.audio_track_checkboxes:
            cb.config(state='disabled')
        self.audio_track_label.config(state='disabled') # Disable the audio track label

        self.listbox.config(state='disabled')
        self.btn_convert.config(state='disabled')
        self.btn_refresh.config(state='disabled')
        
        # Disable mode radio buttons
        if self.mode_frame:
            for rb in self.mode_frame.winfo_children(): 
                if isinstance(rb, ttk.Radiobutton):
                    rb.config(state='disabled')
        
    def _unlock_ui_after_conversion(self):
        # Enable listbox, convert, and refresh buttons
        self.listbox.config(state='normal')
        self.btn_convert.config(state='normal')
        self.btn_refresh.config(state='normal')
        
        # Restore state of other UI elements based on current mode
        self._update_mode_ui() 

    def _start_timer(self):
        self.start_time = time.time()
        self._update_timer()

    def _update_timer(self):
        elapsed = int(time.time() - self.start_time)
        self.lbl_elapsed.config(text=f"경과: {seconds_to_tc(elapsed)}") # Corrected to display elapsed time
        self.timer_id = self.after(1000, self._update_timer) # Update every 1 second

    def _stop_timer(self):
        if self.timer_id:
            self.after_cancel(self.timer_id)
            self.timer_id = None
        self.lbl_elapsed.config(text="경과: 00:00:00") # Reset timer display

    def _open_rename_dialog(self, file_info: Dict):
        current_video_path: Path = file_info['video']
        old_name = current_video_path.stem
        
        # Create a Toplevel window for renaming
        rename_dialog = tk.Toplevel(self)
        rename_dialog.title("파일 이름 바꾸기")
        rename_dialog.transient(self) # Make it appear on top of the main window
        rename_dialog.grab_set()     # Make it modal
        rename_dialog.resizable(False, False)

        # Calculate position to center it on the main window
        self.update_idletasks()
        main_x = self.winfo_x()
        main_y = self.winfo_y()
        main_width = self.winfo_width()
        main_height = self.winfo_height()

        dialog_width = 450 # Slightly increased width
        dialog_height = 130 # Slightly increased height
        dialog_x = main_x + (main_width // 2) - (dialog_width // 2)
        dialog_y = main_y + (main_height // 2) - (dialog_height // 2)
        rename_dialog.geometry(f"{dialog_width}x{dialog_height}+{dialog_x}+{dialog_y}")


        frame = ttk.Frame(rename_dialog, padding=15) # Increased padding
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="새 파일명 입력:", font=("Segoe UI", 10)).pack(pady=(0, 8), anchor="w") # Slightly larger font
        
        new_name_var = tk.StringVar(value=old_name)
        name_entry = ttk.Entry(frame, textvariable=new_name_var, width=60, font=("Segoe UI", 10)) # Increased width and font
        name_entry.pack(fill="x", pady=(0, 15)) # Increased pady below entry
        name_entry.focus_set()

        def on_ok():
            new_name = new_name_var.get().strip()
            if new_name and new_name != old_name:
                try:
                    # Rename video file
                    new_video_path = current_video_path.with_stem(new_name)
                    os.rename(current_video_path, new_video_path)
                    
                    # If PBF exists, rename it too
                    if file_info['pbf'] and file_info['pbf'].exists():
                        current_pbf_path: Path = file_info['pbf']
                        new_pbf_path = current_pbf_path.with_stem(new_name)
                        os.rename(current_pbf_path, new_pbf_path)
                        messagebox.showinfo("이름 변경 완료", f"'{old_name}'을(를) '{new_name}'으로(으로) 변경했습니다. PBF 파일도 함께 변경되었습니다.")
                    else:
                        messagebox.showinfo("이름 변경 완료", f"'{old_name}'을(를) '{new_name}'으로(으로) 변경했습니다.")
                    
                    rename_dialog.destroy()
                    self._refresh_list() # Refresh the list to show new names
                    # Re-select the renamed file if possible
                    for i in range(self.listbox.size()):
                        label = self.listbox.get(i)
                        if new_name in label: # Simple check, can be improved
                            self.listbox.selection_set(i)
                            self.listbox.activate(i)
                            self._on_file_select()
                            break

                except OSError as e:
                    messagebox.showerror("오류", f"파일 이름을 변경하는 중 오류가 발생했습니다: {e}", parent=rename_dialog)
            elif new_name == old_name:
                messagebox.showinfo("알림", "새 파일명이 기존 파일명과 동일합니다. 변경 사항이 없습니다.", parent=rename_dialog)
                rename_dialog.destroy()
            else:
                messagebox.showinfo("알림", "이름 변경이 취소되었습니다.", parent=rename_dialog)
                rename_dialog.destroy()

        # Bind Enter key to on_ok function
        name_entry.bind("<Return>", lambda event=None: on_ok())

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill="x", expand=True)
        ttk.Button(button_frame, text="확인", command=on_ok).pack(side="left", expand=True, padx=5)
        ttk.Button(button_frame, text="취소", command=lambda: rename_dialog.destroy()).pack(side="right", expand=True, padx=5) # Added lambda for on_cancel

        rename_dialog.wait_window() # Wait until the dialog is closed

    def _refresh_list(self):
        """파일 목록을 새로고침합니다. 병렬 처리를 사용하여 성능을 최적화합니다."""
        # 이미 스캔 중이면 중단
        if self.scan_thread and self.scan_thread.is_alive():
            return
            
        sel_indices = self.listbox.curselection()
        selected_label = None
        if sel_indices and sel_indices[0] != 0:
            selected_label = self.listbox.get(sel_indices[0])

        # UI 초기화
        self.listbox.delete(0, tk.END)
        self.files.clear()
        
        # 헤더 추가
        header_text = f"{'파일명':<35s} | {'PBF':<5s} | {'Ch':<5s} | {'Audio':<5s}"
        self.listbox.insert(tk.END, header_text)
        self.listbox.itemconfig(0, {'bg': 'lightgray', 'fg': 'black'})
        
        # 비디오 파일 목록 수집 (이름 순으로 정렬)
        video_files = sorted([path for path in SCRIPT_DIR.iterdir() if path.suffix.lower() in VIDEO_EXTS], key=lambda x: x.name)
        
        if not video_files:
            self.scan_progress_var.set("비디오 파일이 없습니다")
            return
            
        # 병렬 스캔 시작
        self.scan_progress_var.set(f"파일 스캔 중... (0/{len(video_files)})")
        self.scan_thread = threading.Thread(target=self._scan_files_parallel, args=(video_files, selected_label))
        self.scan_thread.start()

    def _scan_files_parallel(self, video_files: List[Path], selected_label: str = None):
        """병렬로 파일들을 스캔합니다."""
        try:
            # 캐시에서 이미 분석된 파일들 확인
            uncached_files = []
            cached_results = {}
            
            for video_path in video_files:
                if video_path in self.file_cache:
                    cached_results[video_path] = self.file_cache[video_path]
                else:
                    uncached_files.append(video_path)
            
            if not uncached_files:
                # 모든 파일이 캐시에 있는 경우, 정렬해서 UI 업데이트
                self.after(0, lambda: self._update_ui_with_sorted_results(cached_results, video_files, selected_label))
                self.after(0, lambda: self.scan_progress_var.set("스캔 완료 (캐시 사용)"))
                return
                
            # 병렬 처리로 새 파일들 분석
            max_workers = min(len(uncached_files), self.cpu_cores.get())
            completed_count = len(cached_results)
            all_results = cached_results.copy()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 작업 제출
                future_to_path = {executor.submit(analyze_single_video, path): path for path in uncached_files}
                
                # 결과 수집
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        self.file_cache[path] = result
                        all_results[path] = result
                        completed_count += 1
                        
                        # 진행 상황 업데이트
                        self.after(0, lambda c=completed_count, t=len(video_files): 
                                 self.scan_progress_var.set(f"파일 스캔 중... ({c}/{t})"))
                        
                    except Exception as e:
                        print(f"Error processing {path}: {e}")
                        completed_count += 1
                        self.after(0, lambda c=completed_count, t=len(video_files): 
                                 self.scan_progress_var.set(f"파일 스캔 중... ({c}/{t})"))
            
            # 모든 분석 완료 후 정렬해서 UI 업데이트
            self.after(0, lambda: self._update_ui_with_sorted_results(all_results, video_files, selected_label))
            self.after(0, lambda: self.scan_progress_var.set("스캔 완료"))
            
        except Exception as e:
            self.after(0, lambda: self.scan_progress_var.set(f"스캔 오류: {e}"))
            print(f"Error in parallel scan: {e}")

    def _update_ui_with_results(self, results: Dict[Path, Dict], selected_label: str = None):
        """캐시된 결과로 UI를 업데이트합니다."""
        for video_path, data in results.items():
            self._add_single_file_to_ui(data)

    def _update_ui_with_sorted_results(self, results: Dict[Path, Dict], video_files: List[Path], selected_label: str = None):
        """정렬된 결과로 UI를 업데이트합니다."""
        # video_files 순서대로 정렬된 결과 생성
        for video_path in video_files:
            if video_path in results:
                self._add_single_file_to_ui(results[video_path])

    def _add_single_file_to_ui(self, data: Dict[str, Any]):
        """단일 파일 데이터를 UI에 추가합니다."""
        video_path = data["video"]
        filename_display = str(video_path.name)
        if len(filename_display) > 35:
            filename_display = filename_display[:32] + "..."
        
        label = (f"{filename_display:<35s} | "
                 f"{str(data['pbf_chaps_count']) if data['pbf_chaps_count'] > 0 else '-':<5s} | "
                 f"{str(data['ff_chaps_count']) if data['ff_chaps_count'] > 0 else '-':<5s} | "
                 f"{str(data['audio_count']) if data['audio_count'] > 0 else '-':<5s}")
        
        self.files[label] = {
            "video": video_path,
            "pbf": data["pbf"],
            "ff_chapters_full": data["ff_chapters_full"],
            "ff_audio_streams": data["ff_audio_streams"]
        }
        self.listbox.insert(tk.END, label)

    def _restore_selection(self, selected_label: str = None):
        """이전 선택을 복원합니다."""
        if selected_label:
            for i in range(1, self.listbox.size()):
                if self.listbox.get(i) == selected_label:
                    self.listbox.selection_set(i)
                    self.listbox.activate(i)
                    break
        else:
            if self.listbox.size() > 1:
                self.listbox.selection_set(1)
                self.listbox.activate(1)
        self._on_file_select()


    def _update_mode_ui(self):
        is_rename = (self.mode.get() == "rename")
        
        # Enable/Disable trim options based on mode
        for rb in self.trim_fr.winfo_children():
            if isinstance(rb, (ttk.Radiobutton, ttk.Checkbutton)): 
                rb.config(state='disabled' if is_rename else 'normal')
        
        # Enable/Disable CPU options based on mode
        for child in self.cpu_fr.winfo_children():
            if isinstance(child, (ttk.Spinbox, ttk.Checkbutton)): 
                child.config(state='disabled' if is_rename else 'normal')

        # Audio options (individual track checkboxes)
        if is_rename:
            self._clear_audio_track_checkboxes()
        else:
            # Re-enable based on whether audio streams were found in the selected file
            sel = self.listbox.curselection()
            if sel and sel[0] != 0:
                info = self.files[self.listbox.get(sel[0])]
                if info['ff_audio_streams']:
                    self.audio_track_label.config(state='normal')
                    for cb in self.audio_track_checkboxes:
                        cb.config(state='normal')
                else:
                    self.audio_track_label.config(text="오디오 트랙 없음", state='disabled')
                    for cb in self.audio_track_checkboxes:
                        cb.config(state='disabled')
            else:
                self.audio_track_label.config(state='disabled')
                for cb in self.audio_track_checkboxes:
                    cb.config(state='disabled')


        # Mode radio buttons are always enabled except during conversion
        if self.mode_frame:
            # Only disable if conversion_thread is active
            if self.conversion_thread and self.conversion_thread.is_alive(): 
                for rb in self.mode_frame.winfo_children():
                    if isinstance(rb, ttk.Radiobutton):
                        rb.config(state='disabled')
                self.btn_convert.config(state='disabled') 
                self.btn_refresh.config(state='disabled') 
                self.listbox.config(state='disabled') 
            else: # Conversion not active, set all to normal
                for rb in self.mode_frame.winfo_children():
                    if isinstance(rb, ttk.Radiobutton):
                        rb.config(state='normal')
                self.btn_convert.config(state='normal')
                self.btn_refresh.config(state='normal')
                self.listbox.config(state='normal')


        # Clear chapter and result texts if in rename mode
        if is_rename:
            self._update_chapter_text([])
            self._update_result_text([])
        else:
            # Re-call _on_file_select to populate chapter/result text if in trim mode
            self._on_file_select()

    # REMOVED: _on_trim_audio_toggle method, as self.trim_audio is removed
    # def _on_trim_audio_toggle(self, *args):
    #     """Callback for when '오디오 트랙 유지' checkbox is toggled."""
    #     if self.trim_audio.get():
    #         self.audio_track_label.config(state='normal')
    #         for cb in self.audio_track_checkboxes:
    #             cb.config(state='normal')
    #     else:
    #         self.audio_track_label.config(state='disabled')
    #         for cb in self.audio_track_checkboxes:
    #             cb.config(state='disabled')
    #             if cb._value.get():
    #                 cb._value.set(False)

    def _clear_audio_track_checkboxes(self):
        """Removes all dynamically created audio track checkboxes and hides the '오디오 트랙 선택:' label."""
        for cb in self.audio_track_checkboxes:
            cb.destroy()
        self.audio_track_checkboxes.clear()
        self.audio_track_vars.clear()
        
        # Hide the label and set text back to original for next display
        self.audio_track_label.config(text="오디오 트랙 선택:", state='disabled') 

    def _update_audio_track_ui(self, audio_streams: List[int]):
        """Dynamically creates checkboxes for detected audio streams."""
        self._clear_audio_track_checkboxes() # Clear existing ones first

        if not audio_streams:
            # No audio streams found, just update the label text to reflect this
            self.audio_track_label.config(text="오디오 트랙 없음", state='normal')
            return
        else:
            # Restore the label text and enable it
            self.audio_track_label.config(text="오디오 트랙 선택:", state='normal')

        # Create new checkboxes
        for stream_idx in audio_streams: # audio_streams now contains 0-based indices
            var = tk.BooleanVar(value=True) # Default to selected
            self.audio_track_vars[stream_idx] = var
            # Use stream_idx + 1 for user-friendly display (1-based index)
            cb = ttk.Checkbutton(self.audio_track_checkbox_container_fr, text=f"{stream_idx + 1}", variable=var)
            cb.pack(side="left", padx=2)
            self.audio_track_checkboxes.append(cb)
            cb._value = var # Store a reference to the BooleanVar in the widget for easy access

        # Individual audio track checkboxes are always enabled if audio streams are present
        for cb in self.audio_track_checkboxes:
            cb.config(state='normal')


    def _on_file_select(self, event=None):
        sel = self.listbox.curselection()
        # If the header row (index 0) is selected, or nothing is selected, clear display and return
        if not sel or sel[0] == 0: 
            self._update_chapter_text([]) 
            self._update_result_text([])
            self._update_audio_track_ui([]) 
            return
            
        info = self.files[self.listbox.get(sel[0])]
        
        # Only update chapter/result text and audio tracks if in trim mode
        if self.mode.get() == "trim":
            strat = self.trim_strategy.get()
            chapters_raw: List[Dict] = []
            
            # Use the full chapter list stored during refresh (cached data)
            if strat in ("chapter", "both"): chapters_raw.extend(info['ff_chapters_full'])
            if strat in ("pbf", "both") and info.get('pbf'): chapters_raw.extend(parse_pbf(info['pbf']))
            
            uniq = {c['start']: c['title'] for c in chapters_raw}
            sorted_chaps = sorted([{'start': s, 'title': t} for s, t in uniq.items()], key=lambda x: x['start'])
            self._update_chapter_text(sorted_chaps)

            # 비디오 길이는 캐시에서 가져옴 (이미 analyze_single_video에서 계산됨)
            video_path = info['video']
            if video_path in self.file_cache and 'duration' in self.file_cache[video_path]:
                duration = self.file_cache[video_path]['duration']
            else:
                # 캐시에 없는 경우에만 계산 (거의 발생하지 않음)
                duration = get_video_duration(video_path)
                    
            results: List[Dict] = []
            trimmed_chaps = sorted_chaps[:-1] if self.between_only.get() and len(sorted_chaps) > 1 else sorted_chaps
            for i, c in enumerate(trimmed_chaps, 1):
                start = c['start']
                end = sorted_chaps[i]['start'] if i < len(sorted_chaps) else duration
                # Filename format: 01_ChapterName.ext, saved in a subfolder named after the original video.
                fname = f"{i:02d}_{sanitize_filename(c['title'])}{info['video'].suffix}"
                results.append({'filename': fname, 'start': seconds_to_tc(start), 'end': seconds_to_tc(end), 'duration': seconds_to_tc(end - start)})
            self._update_result_text(results)
            
            # Update audio track UI for the selected file
            self._update_audio_track_ui(info['ff_audio_streams'])

        else: # If in rename mode, clear the chapter and result texts and audio tracks
            self._update_chapter_text([])
            self._update_result_text([])
            self._update_audio_track_ui([])


    def _update_chapter_text(self, chapters: List[Dict]):
        self.chap_text.config(state='normal')
        self.chap_text.delete('1.0', tk.END)
        for i, c in enumerate(chapters, 1):
            self.chap_text.insert(tk.END, f"{i:02d}. {seconds_to_tc(c['start'])} - {c['title']}\n")
        self.chap_text.config(state='disabled')

    def _update_result_text(self, results: List[Dict]):
        self.res_text.config(state='normal')
        self.res_text.delete('1.0', tk.END)
        for r in results:
            self.res_text.insert(tk.END, f"{r['filename']}: {r['start']} ~ {r['end']} ({r['duration']})\n")
        self.res_text.config(state='disabled')

if __name__ == '__main__':
    # Check for ffmpeg and ffprobe at startup
    ffmpeg_found = is_tool_available(FFMPEG_BIN)
    ffprobe_found = is_tool_available(FFPROBE_BIN)

    if not ffmpeg_found or not ffprobe_found:
        message = (
            "FFmpeg이 설치되어 있지 않거나 시스템 PATH에 등록되어 있지 않습니다.\n"
            "영상 처리 기능을 사용하려면 FFmpeg을 설치해야 합니다.\n\n"
            "설치 방법:\n"
            "1. FFmpeg 공식 웹사이트 방문: https://ffmpeg.org/download.html\n"
            "2. 운영 체제에 맞는 버전을 다운로드하여 설치하거나, 패키지 관리자(예: Windows의 Chocolatey, macOS의 Homebrew, Linux의 apt/dnf)를 사용하여 설치하세요.\n\n"
            "설치 후 PATH 환경 변수에 FFmpeg 실행 파일 경로를 추가해야 할 수 있습니다."
        )
        messagebox.showerror("FFmpeg 필요", message)
        exit() # Exit the application if FFmpeg is not found

    app = VideoToolkitApp()
    app.mainloop()