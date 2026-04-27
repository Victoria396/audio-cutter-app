import shutil
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
from pydub import AudioSegment


st.set_page_config(page_title="Audio Cutter", layout="wide")

WORKDIR = Path("workdir")
UPLOAD_DIR = WORKDIR / "uploads"
CUTS_DIR = WORKDIR / "cuts"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CUTS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_model():
    return WhisperModel(
        "base",
        device="cpu",
        compute_type="int8",
        cpu_threads=8,
        num_workers=1,
    )


def save_uploaded_file(uploaded_file) -> Path:
    path = UPLOAD_DIR / uploaded_file.name
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def get_audio_duration(path: Path) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000


def transcribe(audio_path: Path):
    model = load_model()

    segments, _ = model.transcribe(
        str(audio_path),
        language="ru",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    rows = []
    text_parts = []

    for i, s in enumerate(segments, start=1):
        clean_text = s.text.strip()

        rows.append(
            {
                "cut_here": False,
                "title": f"clip_{i:03d}",
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "text": clean_text,
            }
        )

        text_parts.append(clean_text)

    return rows, " ".join(text_parts)


def safe_filename(name: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name.strip() or "clip"


def make_cuts_from_markers(df: pd.DataFrame, duration: float):
    selected = df[df["cut_here"] == True].copy()

    markers = []

    for _, row in selected.iterrows():
        try:
            start = float(row["start"])
            title = str(row["title"]).strip()
        except Exception:
            continue

        if start < 0 or start >= duration:
            continue

        markers.append(
            {
                "start": round(start, 3),
                "title": title,
            }
        )

    markers = sorted(markers, key=lambda x: x["start"])

    if not markers or markers[0]["start"] > 0:
        markers.insert(
            0,
            {
                "start": 0.0,
                "title": "clip_001",
            },
        )

    unique_markers = []
    seen = set()

    for marker in markers:
        key = round(marker["start"], 3)
        if key not in seen:
            unique_markers.append(marker)
            seen.add(key)

    markers = unique_markers

    cuts = []

    for i, marker in enumerate(markers):
        start = marker["start"]

        if i + 1 < len(markers):
            end = markers[i + 1]["start"]
        else:
            end = duration

        if end <= start:
            continue

        cuts.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "title": f"clip_{i + 1:03d}",
            }
        )

    return cuts


def cut_audio(audio_path: Path, cuts):
    shutil.rmtree(CUTS_DIR, ignore_errors=True)
    CUTS_DIR.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    files = []

    for i, cut in enumerate(cuts, start=1):
        segment = audio[int(cut["start"] * 1000): int(cut["end"] * 1000)]

        filename = f"{i:03d}_{safe_filename(cut['title'])}.mp3"
        output_path = CUTS_DIR / filename

        segment.export(output_path, format="mp3")
        files.append(output_path)

    zip_path = CUTS_DIR / "cuts.zip"

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.name)

    return files, zip_path


def read_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


st.title("Audio → Text → Cutter")

file = st.file_uploader(
    "Загрузи аудио",
    type=["mp3", "wav", "m4a", "aac", "ogg", "flac"],
)

if file:
    path = save_uploaded_file(file)
    duration = get_audio_duration(path)

    st.write(f"Длительность: **{round(duration, 1)} сек**")
    st.audio(str(path))

    if st.button("Расшифровать всё аудио", type="primary"):
        with st.spinner("Транскрибация всего аудио..."):
            rows, text = transcribe(path)

        st.session_state["rows"] = rows
        st.session_state["text"] = text
        st.session_state["cuts_table"] = pd.DataFrame(rows)
        st.session_state.pop("files", None)
        st.session_state.pop("zip", None)

    if "cuts_table" in st.session_state:
        st.subheader("Транскрипт и точки реза")

        st.download_button(
            "Скачать TXT",
            st.session_state["text"].encode("utf-8"),
            file_name="transcript.txt",
            mime="text/plain",
        )

        with st.expander("Полный текст"):
            st.text_area(
                "Текст",
                st.session_state["text"],
                height=220,
            )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Снять выбор со всех"):
                st.session_state["cuts_table"]["cut_here"] = False
                st.rerun()

        with col2:
            if st.button("Поставить точку в начале первой фразы"):
                st.session_state["cuts_table"].loc[0, "cut_here"] = True
                st.rerun()

        st.info(
            "Чекбокс теперь означает: начать новый клип с этой фразы. "
            "Конец клипа будет перед следующей выбранной точкой."
        )

        edited_df = st.data_editor(
            st.session_state["cuts_table"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_order=["cut_here", "start", "end", "text", "title"],
            column_config={
                "cut_here": st.column_config.CheckboxColumn(
                    "Начать клип здесь",
                    default=False,
                ),
                "start": st.column_config.NumberColumn(
                    "Начало",
                    min_value=0.0,
                    max_value=duration,
                    step=0.1,
                    format="%.2f",
                    disabled=True,
                ),
                "end": st.column_config.NumberColumn(
                    "Конец фразы",
                    min_value=0.0,
                    max_value=duration,
                    step=0.1,
                    format="%.2f",
                    disabled=True,
                ),
                "text": st.column_config.TextColumn(
                    "Текст",
                    disabled=True,
                ),
                "title": st.column_config.TextColumn(
                    "Название",
                    disabled=True,
                ),
            },
        )

        st.session_state["cuts_table"] = edited_df

        cuts = make_cuts_from_markers(edited_df, duration)

        st.write(f"Будет создано клипов: **{len(cuts)}**")

        if cuts:
            preview_df = pd.DataFrame(cuts)
            preview_df["duration"] = preview_df["end"] - preview_df["start"]
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Нарезать по выбранным точкам", type="primary"):
            if not cuts:
                st.error("Нет точек реза.")
            else:
                with st.spinner("Нарезка аудио..."):
                    files, zip_path = cut_audio(path, cuts)

                st.session_state["files"] = files
                st.session_state["zip"] = zip_path

    if "files" in st.session_state:
        st.subheader("Предпрослушивание и скачивание")

        for f in st.session_state["files"]:
            st.markdown(f"**{f.name}**")
            st.audio(str(f))

            st.download_button(
                f"Скачать {f.name}",
                read_bytes(f),
                file_name=f.name,
                mime="audio/mp3",
            )

        st.download_button(
            "Скачать всё ZIP",
            read_bytes(st.session_state["zip"]),
            file_name="cuts.zip",
            mime="application/zip",
            type="primary",
        )

else:
    st.info("Загрузи аудиофайл.")