from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import torch
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
YOLOV5_DIR = BASE_DIR / "yolov5"
MODEL_WEIGHTS = BASE_DIR / "weights" / "slow_but_precise.pt"

INPUT_W = 1760
INPUT_H = 992

yolo_path = str(YOLOV5_DIR)
if yolo_path not in sys.path:
    sys.path.insert(0, yolo_path)

if not (YOLOV5_DIR / "models" / "common.py").exists():
    raise FileNotFoundError(f"Nie widzę YOLOv5 w {YOLOV5_DIR} (brak models/common.py)")

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

import ctypes
import traceback


def cuda_preflight() -> None:
    print("\n=== CUDA PREFLIGHT (GUI) ===")
    print("exe:", sys.executable)
    print("torch:", torch.__version__, "| torch.cuda:", getattr(torch.version, "cuda", "n/a"))
    print("torch.__file__:", getattr(torch, "__file__", "n/a"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        ctypes.WinDLL("nvcuda.dll")
        print("nvcuda.dll: OK (driver DLL dostępny)")
    except Exception as e:
        print("nvcuda.dll: BRAK/nieładowalny ->", e)
    try:
        t = torch.ones(1, device="cuda")
        print("torch.ones(..., device='cuda'): OK")
        print("GPU[0]:", torch.cuda.get_device_name(0))
        torch.cuda.synchronize()
    except Exception as e:
        print("BŁĄD inicjalizacji CUDA w tym procesie ->", repr(e))
        traceback.print_exc()
        raise
    print("============================\n")


class YoloVideoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLOv5 – Analiza wideo z boxami (GPU, zgodnie z TRT)")
        self.geometry("760x560")
        self.minsize(720, 520)

        self.video_path: Path | None = None
        self.processing_thread: threading.Thread | None = None
        self.stop_flag = threading.Event()

        self._yolo_backend = None
        self._yolo_device = None
        self._yolo_stride = 32
        self._yolo_names = {}
        self._yolo_fp16 = False

        self._build_ui()

    def _build_ui(self) -> None:
        pad = 10
        frm_top = ttk.Frame(self)
        frm_top.pack(fill=tk.X, padx=pad, pady=(pad, 0))

        ttk.Label(frm_top, text="Plik wideo:").grid(row=0, column=0, sticky=tk.W)
        self.ent_video = ttk.Entry(frm_top)
        self.ent_video.grid(row=0, column=1, sticky=tk.EW, padx=(5, 5))
        ttk.Button(frm_top, text="Wybierz…", command=self.choose_video).grid(row=0, column=2, padx=(0, 5))

        ttk.Label(frm_top, text="Model (na sztywno):").grid(row=1, column=0, sticky=tk.W, pady=(pad // 2, 0))
        self.ent_model = ttk.Entry(frm_top, state="readonly")
        self.ent_model.grid(row=1, column=1, sticky=tk.EW, padx=(5, 5), pady=(pad // 2, 0))
        try:
            self.ent_model.configure(state="normal")
            self.ent_model.delete(0, tk.END)
            self.ent_model.insert(0, Path(MODEL_WEIGHTS).name)
            self.ent_model.configure(state="readonly")
        except Exception:
            pass

        self.var_use_gpu = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm_top, text="Użyj GPU (jeśli dostępne)", variable=self.var_use_gpu).grid(
            row=2, column=1, sticky=tk.W, pady=(pad // 2, 0)
        )

        frm_top.columnconfigure(1, weight=1)

        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill=tk.X, padx=pad, pady=(pad, 0))
        self.btn_start = ttk.Button(frm_actions, text="Start analizy", command=self.start_processing)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop = ttk.Button(frm_actions, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(5, 0))

        frm_prog = ttk.Frame(self)
        frm_prog.pack(fill=tk.X, padx=pad, pady=(pad, 0))
        self.prog = ttk.Progressbar(frm_prog, mode="determinate")
        self.prog.pack(fill=tk.X)

        frm_log = ttk.LabelFrame(self, text="Log")
        frm_log.pack(fill=tk.BOTH, expand=True, padx=pad, pady=(pad, pad))
        self.txt_log = tk.Text(frm_log, height=14, wrap=tk.WORD)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value="Gotowe.")
        ttk.Label(self, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, padx=pad, pady=(0, pad))

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Wybierz plik wideo",
            filetypes=[("Pliki wideo", "*.mp4 *.avi *.mov *.mkv"), ("Wszystkie pliki", "*.*")]
        )
        if path:
            self.video_path = Path(path)
            self.ent_video.delete(0, tk.END)
            self.ent_video.insert(0, str(self.video_path))
            self.log(f"Wybrano wideo: {self.video_path}")

    def start_processing(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            return

        video = self.ent_video.get().strip()
        if not video:
            messagebox.showwarning("Brak wideo", "Wybierz plik wideo.")
            return

        self.video_path = Path(video)
        if not self.video_path.exists():
            messagebox.showerror("Błąd", "Podany plik wideo nie istnieje.")
            return

        out_path = self._default_out_path(self.video_path)

        self.stop_flag.clear()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status.set("Trwa przetwarzanie…")
        self.log("Rozpoczynam analizę…")
        self.prog.config(value=0, maximum=100)

        self.processing_thread = threading.Thread(
            target=self._process_worker,
            args=(self.video_path, out_path, self.var_use_gpu.get()),
            daemon=True,
        )
        self.processing_thread.start()
        self._poll_thread()

    def stop_processing(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            self.stop_flag.set()
            self.log("Zatrzymywanie po bieżącej klatce…")

    def _poll_thread(self) -> None:
        if self.processing_thread and self.processing_thread.is_alive():
            self.after(200, self._poll_thread)
        else:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def log(self, msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        self.txt_log.insert(tk.END, f"[{ts}] {msg}\n")
        self.txt_log.see(tk.END)

    def _default_out_path(self, in_path: Path) -> Path:
        return in_path.with_name(f"{in_path.stem}_yolo.mp4")

    def _log_cuda_info(self) -> None:
        try:
            self.log(f"PyTorch: {torch.__version__} | CUDA dostępne: {torch.cuda.is_available()}")
            self.log(f"CUDA wersja kompilacji Torch: {getattr(torch.version, 'cuda', 'n/a')}")
            self.log(f"GPU count: {torch.cuda.device_count()}")
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                for i in range(torch.cuda.device_count()):
                    self.log(f"GPU[{i}]: {torch.cuda.get_device_name(i)}")
        except Exception as e:
            self.log(f"CUDA info błąd: {e}")

    def _load_model(self, use_gpu: bool):
        self._log_cuda_info()

        if use_gpu:
            cuda_preflight()
            device = select_device("cuda:0")
            torch.backends.cudnn.benchmark = True
            fp16 = True
        else:
            device = select_device("cpu")
            fp16 = False

        if not Path(MODEL_WEIGHTS).exists():
            raise FileNotFoundError(f"Nie znaleziono wag: {MODEL_WEIGHTS}")

        model = DetectMultiBackend(MODEL_WEIGHTS, device=device, dnn=False, data=None, fp16=fp16)

        self._yolo_backend = "dmb"
        self._yolo_device = device
        self._yolo_stride = int(model.stride)
        self._yolo_names = model.names
        self._yolo_fp16 = bool(getattr(model, "fp16", False) and (device.type == "cuda"))

        if getattr(model, "pt", False) and self._yolo_fp16:
            model.model.half().eval()

        if device.type == "cuda":
            dummy = torch.zeros(1, 3, INPUT_H, INPUT_W, device=device)
            dummy = dummy.half() if self._yolo_fp16 else dummy.float()
            _ = model(dummy)
            torch.cuda.synchronize()

        backend = "TensorRT" if getattr(model, "engine", False) else ("PyTorch" if getattr(model, "pt", False) else "Other")
        self.log(f"YOLO: DetectMultiBackend | backend: {backend} | device: {device} | fp16: {self._yolo_fp16} | stride: {self._yolo_stride}")
        return model

    def _process_worker(self, video_path: Path, out_path: Path, use_gpu: bool) -> None:
        try:
            model = self._load_model(use_gpu)

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError("Nie można otworzyć pliku wideo.")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            in_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc = in_fourcc if in_fourcc != 0 else cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Nie można utworzyć pliku wyjściowego.")

            self.log(f"Rozdzielczość wejścia: {width}x{height}, FPS: {fps:.2f}, klatek: {total}")
            self.log(f"YOLO input (letterbox): {INPUT_W}x{INPUT_H}; wyjście: oryginalna rozdzielczość")
            self.log(f"Zapis do: {out_path}")

            if total > 0:
                self.after(0, lambda: self.prog.config(mode="determinate", maximum=100, value=0))
            else:
                self.after(0, lambda: (self.prog.config(mode="indeterminate"), self.prog.start(12)))

            idx = 0
            last_pct = -1
            names = getattr(self, "_yolo_names", {})

            while True:
                if self.stop_flag.is_set():
                    self.log("Przetwarzanie przerwane przez użytkownika.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                det = None
                if getattr(self, "_yolo_backend", "hub") == "dmb" and letterbox is not None:
                    im0 = frame
                    img = letterbox(im0, new_shape=(INPUT_H, INPUT_W), stride=self._yolo_stride, auto=False)[0]
                    img = img[:, :, ::-1].transpose(2, 0, 1)
                    img = np.ascontiguousarray(img)

                    im = torch.from_numpy(img).to(self._yolo_device)
                    im = (im.half() if self._yolo_fp16 else im.float()) / 255.0
                    im = im.unsqueeze(0)

                    if im.shape[-2:] != (INPUT_H, INPUT_W):
                        raise RuntimeError(f"input size {tuple(im.shape)} not equal to model size (1,3,{INPUT_H},{INPUT_W})")

                    with torch.inference_mode():
                        pred = model(im)
                    det_t = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, agnostic=True, max_det=300)[0]
                    if det_t is not None and len(det_t):
                        det_t[:, :4] = scale_boxes(im.shape[2:], det_t[:, :4], im0.shape).round()
                        det = det_t.detach().to("cpu").numpy()
                else:
                    frame_resized = cv2.resize(frame, (INPUT_W, INPUT_H), interpolation=cv2.INTER_LINEAR)
                    results = model(frame_resized)
                    det = results.xyxy[0].detach().cpu().numpy() if hasattr(results, "xyxy") else None

                if det is not None and len(det):
                    def _cls_name(names_, cls_id):
                        if isinstance(names_, (list, tuple)):
                            idxn = int(cls_id)
                            return names_[idxn] if 0 <= idxn < len(names_) else str(idxn)
                        return str(names_.get(int(cls_id), int(cls_id)))

                    for *xyxy, conf, cls_id in det:
                        x1, y1, x2, y2 = map(int, xyxy)
                        label = _cls_name(names, cls_id)
                        text = f"{label} {float(conf):.2f}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        y_text = max(y1, th + 6)
                        cv2.rectangle(frame, (x1, y_text - th - 6), (x1 + tw + 4, y_text), (0, 255, 0), -1)
                        cv2.putText(
                            frame,
                            text,
                            (x1 + 2, y_text - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            1,
                            cv2.LINE_AA,
                        )

                writer.write(frame)

                idx += 1
                if total > 0:
                    pct = int((idx / total) * 100)
                    if pct != last_pct:
                        self._set_progress(pct)
                        last_pct = pct

            cap.release()
            writer.release()
            self.after(0, lambda: self.prog.stop())

            if not self.stop_flag.is_set():
                if total > 0:
                    self._set_progress(100)
                self.status.set("Zakończono.")
                self.log("Gotowe.")
                self.log(f"Plik wynikowy: {out_path}")
            else:
                self.status.set("Przerwano.")

        except Exception as e:
            self.status.set("Błąd.")
            self.log(f"Błąd: {e}")

        finally:
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)

    def _set_progress(self, value: int) -> None:
        self.after(0, lambda: self.prog.config(value=value))


def main() -> None:
    if sys.version_info < (3, 8):
        messagebox.showwarning("Python", "Zalecany Python 3.8 lub nowszy.")
    app = YoloVideoApp()
    app.mainloop()


if __name__ == "__main__":
    main()
