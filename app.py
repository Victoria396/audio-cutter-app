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
CHUNKS_DIR = WORKDIR / "chunks"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CUTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_model():
    return WhisperModel(
        "tiny",
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


def split_audio_to_chunks(audio_path: Path, chunk_minutes: int = 10):
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []

    for i, start_ms in enumerate(range(0, len(audio), chunk_ms), start=1):
        end_ms = min(start_ms + chunk_ms, len(audio))
        chunk = audio[start_ms:end_ms]

        chunk_path = CHUNKS_DIR / f"chunk_{i:03d}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(
            {
                "path": chunk_path,
                "offset_seconds": start_ms / 1000,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )

    return chunks


def transcribe_chunk(chunk_path: Path, offset_seconds: float):
    model = load_model()

    segments, _ = model.transcribe(
        str(chunk_path),
        language="ru",
        beam_size=1,
        best_of=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    rows = []

    for s in segments:
        clean_text = s.text.strip()

        if not clean_text:
            continue

        rows.append(
            {
                "cut_here": False,
                "start": round(s.start + offset_seconds, 2),
                "end": round(s.end + offset_seconds, 2),
                "text": clean_text,
            }
        )

    return rows


def transcribe_long_audio(audio_path: Path, chunk_minutes: int = 10):
    chunks = split_audio_to_chunks(audio_path, chunk_minutes=chunk_minutes)

    all_rows = []
    progress = st.progress(0)
    status = st.empty()

    for i, chunk in enumerate(chunks, start=1):
        status.write(
            f"Транскрибация части {i} из {len(chunks)} "
            f"({round(chunk['offset_seconds'] / 60, 1)} мин)"
        )

        rows = transcribe_chunk(
            chunk_path=chunk["path"],
            offset_seconds=chunk["offset_seconds"],
        )

        all_rows.extend(rows)
        progress.progress(i / len(chunks))

    status.write("Транскрибация завершена.")

    full_text = " ".join(row["text"] for row in all_rows)

    return all_rows, full_text


def safe_filename(name: str) -> str:
    bad_chars = '<>:"/\\|?*'

    for ch in bad_chars:
        name = name.replace(ch, "_")

    return name.strip() or "clip"


def make_segments_from_markers(df: pd.DataFrame, duration: float):
    selected = df[df["cut_here"] == True].copy()

    markers = []

    for _, row in selected.iterrows():
        try:
            start = float(row["start"])
        except Exception:
            continue

        if 0 <= start < duration:
            markers.append(start)

    markers = sorted(set(round(m, 3) for m in markers))

    if not markers or markers[0] > 0:
        markers.insert(0, 0.0)

    segments = []

    for i, start in enumerate(markers):
        if i + 1 < len(markers):
            end = markers[i + 1]
        else:
            end = duration

        if end <= start:
            continue

        segments.append(
            {
                "title": f"clip_{i + 1:03d}",
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    return pd.DataFrame(segments)


def normalize_manual_segments(df: pd.DataFrame, duration: float):
    segments = []

    for i, row in df.iterrows():
        try:
            title = str(row.get("title", "")).strip()
            start = float(row["start"])
            end = float(row["end"])
        except Exception:
            continue

        if not title or title.lower() == "nan":
            title = f"clip_{len(segments) + 1:03d}"

        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))

        if end <= start:
            continue

        segments.append(
            {
                "title": safe_filename(title),
                "start": round(start, 3),
                "end": round(end, 3),
            }
        )

    segments = sorted(segments, key=lambda x: x["start"])
    return segments


def cut_audio(audio_path: Path, segments):
    shutil.rmtree(CUTS_DIR, ignore_errors=True)
    CUTS_DIR.mkdir(parents=True, exist_ok=True)

    audio = AudioSegment.from_file(audio_path)
    files = []

    for i, segment_data in enumerate(segments, start=1):
        start_ms = int(segment_data["start"] * 1000)
        end_ms = int(segment_data["end"] * 1000)

        segment = audio[start_ms:end_ms]

        filename = f"{i:03d}_{safe_filename(segment_data['title'])}.mp3"
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

    st.write(f"Длительность: **{round(duration / 60, 1)} мин**")
    st.audio(str(path))

    chunk_minutes = st.selectbox(
        "Размер частей для транскрибации",
        [5, 10, 15, 20],
        index=1,
        help="Для длинных файлов лучше 10 минут.",
    )

    if st.button("Расшифровать всё аудио по частям", type="primary"):
        st.session_state.pop("files", None)
        st.session_state.pop("zip", None)
        st.session_state.pop("cuts_table_editor", None)
        st.session_state.pop("manual_segments_editor", None)
        st.session_state.pop("manual_segments", None)

        with st.spinner("Идёт транскрибация..."):
            rows, text = transcribe_long_audio(
                audio_path=path,
                chunk_minutes=chunk_minutes,
            )

        st.session_state["text"] = text
        st.session_state["cuts_table"] = pd.DataFrame(rows)

    if "cuts_table" in st.session_state:
        st.subheader("1. Транскрипт и точки начала клипов")

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
                st.session_state.pop("cuts_table_editor", None)
                st.session_state.pop("manual_segments", None)
                st.session_state.pop("manual_segments_editor", None)
                st.rerun()

        with col2:
            if st.button("Поставить точку в начале первой фразы"):
                if len(st.session_state["cuts_table"]) > 0:
                    st.session_state["cuts_table"].loc[
                        st.session_state["cuts_table"].index[0],
                        "cut_here",
                    ] = True
                st.session_state.pop("cuts_table_editor", None)
                st.session_state.pop("manual_segments", None)
                st.session_state.pop("manual_segments_editor", None)
                st.rerun()

        st.info(
            "Чекбокс означает: начать новый клип с этой фразы. "
            "После выбора точек ниже появится таблица итоговых клипов, где можно вручную менять начало, конец и название."
        )

        edited_df = st.data_editor(
            st.session_state["cuts_table"],
            key="cuts_table_editor",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_order=["cut_here", "start", "end", "text"],
            column_config={
                "cut_here": st.column_config.CheckboxColumn(
                    "Начать клип здесь",
                    default=False,
                ),
                "start": st.column_config.NumberColumn(
                    "Начало фразы",
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
            },
        )

        st.session_state["cuts_table"] = edited_df.copy()

        generated_segments_df = make_segments_from_markers(edited_df, duration)

        st.subheader("2. Итоговые клипы — можно редактировать вручную")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Сформировать клипы из выбранных точек"):
                st.session_state["manual_segments"] = generated_segments_df.copy()
                st.session_state.pop("manual_segments_editor", None)
                st.rerun()

        with col_b:
            if st.button("Создать один клип на весь файл"):
                st.session_state["manual_segments"] = pd.DataFrame(
                    [{"title": "full_audio", "start": 0.0, "end": round(duration, 3)}]
                )
                st.session_state.pop("manual_segments_editor", None)
                st.rerun()

        if "manual_segments" not in st.session_state:
            st.session_state["manual_segments"] = generated_segments_df.copy()

        manual_df = st.data_editor(
            st.session_state["manual_segments"],
            key="manual_segments_editor",
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_order=["title", "start", "end"],
            column_config={
                "title": st.column_config.TextColumn(
                    "Название файла",
                    required=True,
                ),
                "start": st.column_config.NumberColumn(
                    "Начало клипа",
                    min_value=0.0,
                    max_value=duration,
                    step=0.1,
                    format="%.3f",
                ),
                "end": st.column_config.NumberColumn(
                    "Конец клипа",
                    min_value=0.0,
                    max_value=duration,
                    step=0.1,
                    format="%.3f",
                ),
            },
        )

        st.session_state["manual_segments"] = manual_df.copy()

        segments = normalize_manual_segments(manual_df, duration)

        st.write(f"Валидных клипов: **{len(segments)}**")

        if segments:
            preview_df = pd.DataFrame(segments)
            preview_df["duration"] = preview_df["end"] - preview_df["start"]
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Нарезать по ручной таблице", type="primary"):
            if not segments:
                st.error("Нет валидных клипов. Проверь start/end.")
            else:
                with st.spinner("Нарезка аудио..."):
                    files, zip_path = cut_audio(path, segments)

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
