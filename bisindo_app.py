import time
import cv2
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="BISINDO",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False


# ── Konstanta ─────────────────────────────────────────────────────────────────
FPS = 8
CONF = 0.50
IOU = 0.30
CONFIRM_SEC = 2


# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_model(path: str):
    if not YOLO_OK:
        return None
    import os
    return YOLO(path) if os.path.exists(path) else None


def safe_infer(model, frame, conf, iou):
    try:
        results = model(frame, verbose=False, conf=conf, iou=iou, agnostic_nms=False)
        if not results:
            return None
        return results[0]
    except Exception as e:
        st.error(f"Inferensi gagal: {e}")
        return None


# ── Parse raw boxes ───────────────────────────────────────────────────────────
def parse_boxes(boxes, names: dict) -> list[dict]:
    if boxes is None:
        return []
    result = []
    try:
        for box in boxes:
            cls = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0])
            conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0])
            xy = box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else list(box.xyxy[0])
            x1, y1, x2, y2 = map(int, xy)
            result.append(
                {
                    "label": names.get(cls, str(cls)),
                    "conf": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                }
            )
    except Exception:
        return []
    return result


# ── Merge boxes dengan label sama → enclosing box ────────────────────────────
def smart_merge(detections: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for d in detections:
        groups.setdefault(d["label"], []).append(d)

    result = []
    for label, group in groups.items():
        if not group:
            continue
        result.append(
            {
                "label": label,
                "conf": max(d["conf"] for d in group),
                "x1": min(d["x1"] for d in group),
                "y1": min(d["y1"] for d in group),
                "x2": max(d["x2"] for d in group),
                "y2": max(d["y2"] for d in group),
                "merged": len(group) >= 2,
                "count": len(group),
            }
        )
    return result


# ── Label dominan ─────────────────────────────────────────────────────────────
def get_dominant_label(detections: list[dict]) -> str:
    if not detections:
        return ""
    label_confs: dict[str, list[float]] = {}
    for d in detections:
        label = d["label"]
        if label:
            label_confs.setdefault(label, []).append(d["conf"])
    if not label_confs:
        return ""
    return max(label_confs, key=lambda l: sum(label_confs[l]) / len(label_confs[l]))


# ── Draw ──────────────────────────────────────────────────────────────────────
COLOR_MERGED = (52, 168, 83)
COLOR_SINGLE = (66, 133, 244)


def draw(frame: np.ndarray, detections: list[dict]) -> np.ndarray:
    out = frame.copy()
    if out is None or out.size == 0:
        return out

    h, w = out.shape[:2]

    for d in detections:
        x1 = max(0, min(d["x1"], w - 1))
        y1 = max(0, min(d["y1"], h - 1))
        x2 = max(0, min(d["x2"], w - 1))
        y2 = max(0, min(d["y2"], h - 1))

        color = COLOR_MERGED if d.get("merged") else COLOR_SINGLE
        suffix = " (2 tangan)" if d.get("merged") else ""
        text = f"{d['label']}{suffix}: {d['conf']*100:.1f}%"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if d.get("merged"):
            clen = 15
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                dx = clen if cx == x1 else -clen
                dy = clen if cy == y1 else -clen
                cv2.line(out, (cx, cy), (cx + dx, cy), color, 4)
                cv2.line(out, (cx, cy), (cx, cy + dy), color, 4)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(th + 8, y1)
        cv2.rectangle(out, (x1, y_text - th - 8), (x1 + tw + 4, y_text), color, -1)
        cv2.putText(
            out,
            text,
            (x1 + 2, y_text - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60:02d}:{s % 60:02d}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🤟 BISINDO")
page = st.sidebar.radio(
    "Navigasi",
    ["Home", "🔍 YOLOv8m", "🔍 YOLOv11m"],
    label_visibility="collapsed",
)


def init_state(model_name: str):
    run_key = f"{model_name}_run"
    sent_key = f"{model_name}_sentence"
    word_key = f"{model_name}_word"
    time_key = f"{model_name}_word_time"
    start_key = f"{model_name}_start_time"
    clear_key = f"{model_name}_clear_req"

    defaults = {
        run_key: False,
        sent_key: [],
        word_key: "",
        time_key: 0.0,
        start_key: 0.0,
        clear_key: False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    return run_key, sent_key, word_key, time_key, start_key, clear_key


# ═══════════════════════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Home":
    st.title("Identifikasi Bahasa Isyarat Indonesia (BISINDO)")
    st.markdown(
        """
Streamlit ini mendeteksi gerakan tangan **BISINDO** secara real-time dengan menggunakan model **YOLOv8m** dan **YOLOv11m**.

**Cara penggunaan:**
1. Pilih model di sidebar (YOLOv8m atau YOLOv11m)
2. Tekan **START** untuk menyalakan kamera
3. Gerakan yang stabil selama **2 detik** otomatis ditambahkan ke kalimat
4. Jika 2 tangan terdeteksi dengan kosakata sama, maka akan digabung menjadi 1 bounding box besar dengan label "(2 tangan)"
"""
    )

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL PAGE
# ═══════════════════════════════════════════════════════════════════════════════
else:
    model_name = "YOLOv8m" if "v8" in page else "YOLOv11m"
    model_file = "yolov8_best.pt" if "v8" in page else "yolov11_best.pt"

    st.title(f"Identifikasi BISINDO — {model_name}")

    # st.sidebar.markdown("---")
    # debug_mode = st.sidebar.checkbox(
    #     "🐛 Debug Mode",
    #     help="Tampilkan info BB sebelum dan sesudah merge.",
    # )

    run_key, sent_key, word_key, time_key, start_key, clear_key = init_state(model_name)
    
    # ── Fungsi callback & helper ──────────────────────────────────────────────
    def show_sentence():
        txt = " ".join(st.session_state[sent_key])
        sentence_ph.info(txt if txt else "_Belum ada kata…_")

    def show_timer():
        start = st.session_state.get(start_key, 0.0)
        if not st.session_state.get(run_key) or start == 0.0:
            timer_ph.caption("_Sesi belum dimulai_")
            return
        elapsed = time.time() - start
        timer_ph.markdown(f"### 🕐 {fmt_duration(elapsed)}")

    def on_start():
        st.session_state[run_key] = True
        st.session_state[start_key] = time.time()

    def on_stop():
        st.session_state[run_key] = False

    def on_clear_sentence():
        st.session_state[sent_key] = []
        st.session_state[word_key] = ""
        st.session_state[time_key] = 0.0
        st.session_state[clear_key] = True

    # ── Tombol START / STOP / Hapus — tepat di bawah judul ───────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.button(
            "▶ START",
            disabled=st.session_state[run_key],
            use_container_width=True,
            on_click=on_start,
        )
    with col2:
        st.button(
            "⏹ STOP",
            disabled=not st.session_state[run_key],
            use_container_width=True,
            on_click=on_stop,
        )
    with col3:
        st.button(
            "🗑 Hapus Kalimat",
            use_container_width=True,
            on_click=on_clear_sentence,
        )

    # ── Layout kamera (kiri) + info panel (kanan) ─────────────────────────────
    cam_col, info_col = st.columns([3, 1])

    with cam_col:
        frame_ph = st.empty()

    with info_col:
        st.markdown("**Status Deteksi:**")
        status_ph = st.empty()

        st.markdown("---")
        st.markdown("**⏱ Waktu Sesi:**")
        timer_ph = st.empty()

        st.markdown("---")
        st.markdown("**Kalimat terdeteksi:**")
        # sentence_ph sudah dibuat di atas; tampilkan isinya di sini via info_col
        sentence_ph = st.empty()

        # if debug_mode:
        #     st.markdown("**Debug — sebelum merge:**")
        #     debug_raw_ph = st.empty()
        #     st.markdown("**Debug — sesudah merge:**")
        #     debug_merged_ph = st.empty()

    show_sentence()
    show_timer()

    if not st.session_state[run_key]:
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            "Tekan START",
            (80, 185),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (100, 100, 100),
            2,
        )
        frame_ph.image(blank, channels="BGR", width=640,)
        status_ph.empty()
        st.stop()

    model = load_model(model_file)
    if model is None:
        st.error(f"Model `{model_file}` tidak ditemukan atau `ultralytics` belum terinstal.")
        st.session_state[run_key] = False
        st.stop()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Webcam tidak dapat dibuka.")
        st.session_state[run_key] = False
        st.stop()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_delay = 1.0 / FPS

    try:
        while st.session_state.get(run_key, False):
            t0 = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                status_ph.warning("Frame kamera gagal dibaca")
                time.sleep(0.1)
                continue

            if st.session_state.get(clear_key):
                st.session_state[sent_key] = []
                st.session_state[word_key] = ""
                st.session_state[time_key] = 0.0
                st.session_state[clear_key] = False
                show_sentence()

            r = safe_infer(model, frame, CONF, IOU)
            if r is None:
                frame_ph.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                status_ph.warning("Tidak ada hasil deteksi atau error inferensi")
                continue

            detections_raw = parse_boxes(r.boxes, getattr(r, "names", {}))
            detections_merged = smart_merge(detections_raw)

            annotated = draw(frame, detections_merged)
            frame_ph.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )

            if detections_merged:
                lines = []
                for d in detections_merged:
                    icon = "🤲" if d.get("merged") else "🖐"
                    lines.append(f"**{d['label']}** {icon} {d['conf']*100:.1f}%")
                status_ph.success("\n\n".join(lines))
            else:
                status_ph.warning("Tidak ada gerakan terdeteksi")

            show_timer()

            top_label = get_dominant_label(detections_raw)
            now = time.time()

            if top_label and top_label != st.session_state[word_key]:
                st.session_state[word_key] = top_label
                st.session_state[time_key] = now
            elif top_label and (now - st.session_state[time_key]) >= CONFIRM_SEC:
                sentence = st.session_state[sent_key]
                if not sentence or sentence[-1] != top_label:
                    sentence.append(top_label)
                    show_sentence()
                st.session_state[time_key] = now

            elapsed = time.time() - t0
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    finally:
        cap.release()