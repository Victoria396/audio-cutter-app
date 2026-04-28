import shutil
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError


st.set_page_config(page_title="Audio Cutter", layout="wide")

WORKDIR = Path("workdir")
UPLOAD_DIR = WORKDIR / "uploads"
CUTS_DIR = WORKDIR / "cuts"
CHUNKS_DIR = WORKDIR / "chunks"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CUTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


def init_state():
    st.session_state.setdefault("cuts_editor_version", 0)
    st.session_state.setdefault("manual_editor_version", 0)


def bump_cuts_editor():
    st.session_state["cuts_editor_version"] += 1


def bump_manual_editor():
    st.session_state["manual_editor_version"] += 1


init_state()


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
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000
    except CouldntDecodeError as e:
        raise ValueError(
            "FFmpeg не смог прочитать файл. Файл повреждён или не является настоящим аудио."
        ) from e


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


def apply_editor_changes(base_df: pd.DataFrame, editor_key: str) -> pd.DataFrame:
    """
    Применяет изменения st.data_editor к базовому DataFrame.

    Важно:
    - base_df не перезаписывается результатом data_editor на каждом rerun;
    - изменения читаются из st.session_state[editor_key];
    - это убирает баг "изменение сохраняется только со второго раза".
    """
    df = base_df.copy().reset_index(drop=True)

    state = st.session_state.get(editor_key)

    if not isinstance(state, dict):
        return df

    edited_rows = state.get("edited_rows", {})
    added_rows = state.get("added_rows", [])
    deleted_rows = state.get("deleted_rows", [])

    for row_idx in sorted(deleted_rows, reverse=True):
        try:
            row_idx = int(row_idx)
        except Exception:
            continue

        if 0 <= row_idx < len(df):
            df = df.drop(df.index[row_idx]).reset_index(drop=True)

    for row_idx, changes in edited_rows.items():
        try:
            row_idx = int(row_idx)
        except Exception:
            continue

        if 0 <= row_idx < len(df):
            for col, value in changes.items():
                if col in df.columns:
                    df.at[row_idx, col] = value

    if added_rows:
        added_df = pd.DataFrame(added_rows)

        for col in df.columns:
            if col not in added_df.columns:
                added_df[col] = None

        added_df = added_df[df.columns]
        df = pd.concat([df, added_df], ignore_index=True)

    return df.reset_index(drop=True)


def make_segments_from_markers(df: pd.DataFrame, duration: float):
    if df.empty or "cut_here" not in df.columns:
        return pd.DataFrame(columns=["title", "start", "end"])

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
        end = markers[i + 1] if i + 1 < len(markers) else duration

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

    if df.empty:
        return segments

    for _, row in df.iterrows():
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

    return sorted(segments, key=lambda x: x["start"])


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

    try:
        duration = get_audio_duration(path)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.write(f"Длительность: **{round(duration / 60, 1)} мин**")
    st.audio(str(path))

    chunk_minutes = st.selectbox(
        "Размер частей для транскрибации",
        [5, 10, 15, 20],
        index=1,
        help="Для длинных файлов лучше 10 минут.",
    )

    if st.button("Расшифровать всё аудио по частям", type="primary"):
        for key in [
            "files",
            "zip",
            "cuts_table_base",
            "manual_segments_base",
            "text",
        ]:
            st.session_state.pop(key, None)

        bump_cuts_editor()
        bump_manual_editor()

        with st.spinner("Идёт транскрибация..."):
            rows, text = transcribe_long_audio(
                audio_path=path,
                chunk_minutes=chunk_minutes,
            )

        cuts_table = pd.DataFrame(rows)

        if not cuts_table.empty:
            cuts_table["cut_here"] = cuts_table["cut_here"].astype(bool)
            cuts_table["start"] = cuts_table["start"].astype(float)
            cuts_table["end"] = cuts_table["end"].astype(float)
            cuts_table["text"] = cuts_table["text"].astype(str)

        st.session_state["text"] = text
        st.session_state["cuts_table_base"] = cuts_table

        bump_cuts_editor()
        bump_manual_editor()
        st.rerun()

    if "cuts_table_base" in st.session_state:
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
                df = st.session_state["cuts_table_base"].copy()
                df["cut_here"] = False
                st.session_state["cuts_table_base"] = df
                st.session_state.pop("manual_segments_base", None)

                bump_cuts_editor()
                bump_manual_editor()
                st.rerun()

        with col2:
            if st.button("Поставить точку в начале первой фразы"):
                df = st.session_state["cuts_table_base"].copy()

                if len(df) > 0:
                    df.loc[df.index[0], "cut_here"] = True

                st.session_state["cuts_table_base"] = df
                st.session_state.pop("manual_segments_base", None)

                bump_cuts_editor()
                bump_manual_editor()
                st.rerun()

        st.info(
            "Чекбокс означает: начать новый клип с этой фразы. "
            "После выбора точек ниже появится таблица итоговых клипов, где можно вручную менять начало, конец и название."
        )

        cuts_key = f"cuts_table_editor_{st.session_state['cuts_editor_version']}"

        st.data_editor(
            st.session_state["cuts_table_base"],
            key=cuts_key,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_order=["cut_here", "start", "end", "text"],
            disabled=["start", "end", "text"],
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
                ),
                "end": st.column_config.NumberColumn(
                    "Конец фразы",
                    min_value=0.0,
                    max_value=duration,
                    step=0.1,
                    format="%.2f",
                ),
                "text": st.column_config.TextColumn("Текст"),
            },
        )

        current_cuts_df = apply_editor_changes(
            st.session_state["cuts_table_base"],
            cuts_key,
        )

        if not current_cuts_df.empty:
            current_cuts_df["cut_here"] = (
                current_cuts_df["cut_here"].fillna(False).astype(bool)
            )

        generated_segments_df = make_segments_from_markers(current_cuts_df, duration)

        st.subheader("2. Итоговые клипы — можно редактировать вручную")

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("Сформировать клипы из выбранных точек"):
                st.session_state["cuts_table_base"] = current_cuts_df.copy()
                st.session_state["manual_segments_base"] = generated_segments_df.copy()

                bump_cuts_editor()
                bump_manual_editor()
                st.rerun()

        with col_b:
            if st.button("Создать один клип на весь файл"):
                st.session_state["cuts_table_base"] = current_cuts_df.copy()
                st.session_state["manual_segments_base"] = pd.DataFrame(
                    [
                        {
                            "title": "full_audio",
                            "start": 0.0,
                            "end": round(duration, 3),
                        }
                    ]
                )

                bump_cuts_editor()
                bump_manual_editor()
                st.rerun()

        if "manual_segments_base" not in st.session_state:
            st.session_state["manual_segments_base"] = generated_segments_df.copy()

        manual_key = f"manual_segments_editor_{st.session_state['manual_editor_version']}"

        st.data_editor(
            st.session_state["manual_segments_base"],
            key=manual_key,
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

        current_manual_df = apply_editor_changes(
            st.session_state["manual_segments_base"],
            manual_key,
        )

        segments = normalize_manual_segments(current_manual_df, duration)

        st.write(f"Валидных клипов: **{len(segments)}**")

        if segments:
            preview_df = pd.DataFrame(segments)
            preview_df["duration"] = preview_df["end"] - preview_df["start"]
            st.dataframe(preview_df, use_container_width=True)

        if st.button("Нарезать по ручной таблице", type="primary"):
            if not segments:
                st.error("Нет валидных клипов. Проверь start/end.")
            else:
                st.session_state["cuts_table_base"] = current_cuts_df.copy()
                st.session_state["manual_segments_base"] = current_manual_df.copy()

                bump_cuts_editor()
                bump_manual_editor()

                with st.spinner("Нарезка аудио..."):
                    files, zip_path = cut_audio(path, segments)

                st.session_state["files"] = files
                st.session_state["zip"] = zip_path
                st.rerun()

    if "files" in st.session_state:
        st.subheader("Предпрослушивание и скачивание")

        for f in st.session_state["files"]:
            st.markdown(f"**{f.name}**")
            st.audio(str(f))

            st.download_button(
                f"Скачать {f.name}",
                read_bytes(f),
                file_name=f.name,
                mime="audio/mpeg",
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