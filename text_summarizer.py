"""
Text Summarizer
----------------
A desktop GUI application that solves a real-world problem: readers lack
time to go through long articles. Paste any article and get a clean,
short summary in seconds.

Tech stack:
    - GUI      : PyQt5
    - NLP      : HuggingFace Transformers (summarization pipeline)
    - Algorithm: Abstractive summarization using the DistilBART model
                 (sshleifer/distilbart-cnn-12-6) - a distilled version
                 of BART fine-tuned on CNN/DailyMail, fast + accurate.

Run:
    python text_summarizer.py
"""

import sys
import threading

# IMPORTANT: torch must be imported before PyQt5 on Windows. PyQt5 changes
# the process's DLL search path when it loads, which can cause Windows to
# fail to locate torch's own C++ DLLs (e.g. c10.dll) if torch is imported
# afterwards. Importing torch first avoids "WinError 1114" crashes.
try:
    import torch  # noqa: F401
except ImportError:
    pass

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIcon, QTextCursor
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit,
    QPushButton, QComboBox, QFrame, QGraphicsDropShadowEffect, QMessageBox,
    QSizePolicy
)
from PyQt5.QtGui import QColor


# --------------------------------------------------------------------------
# Background worker: keeps the GUI responsive while the model loads/runs.
# --------------------------------------------------------------------------
class SummarizerWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    _model = None
    _tokenizer = None
    _lock = threading.Lock()
    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

    def __init__(self, text, length_mode="Medium"):
        super().__init__()
        self.text = text
        self.length_mode = length_mode

    def _get_model(self):
        # Load the model + tokenizer once and cache them on the class, so
        # repeated summarizations after the first one are fast.
        #
        # NOTE: we load the model and tokenizer directly (instead of using
        # transformers.pipeline("summarization", ...)) because some newer
        # transformers releases have removed the "summarization" shortcut
        # from the pipeline task registry. Loading the model class directly
        # works across versions.
        with SummarizerWorker._lock:
            if SummarizerWorker._model is None:
                self.status.emit("Loading summarization model (first run only)...")
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                SummarizerWorker._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                SummarizerWorker._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_NAME)
            return SummarizerWorker._model, SummarizerWorker._tokenizer

    def _summarize_chunk(self, model, tokenizer, chunk, min_len, max_len):
        import torch
        inputs = tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=1024
        )
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
            )
        return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def run(self):
        try:
            model, tokenizer = self._get_model()
            self.status.emit("Summarizing...")

            length_settings = {
                "Short": (30, 60),
                "Medium": (60, 130),
                "Long": (120, 220),
            }
            min_len, max_len = length_settings.get(self.length_mode, (60, 130))

            # The model has an input token limit, so long articles are
            # chunked and summarized piece by piece, then stitched together.
            chunks = self._chunk_text(self.text, max_chars=3000)
            summaries = []
            for i, chunk in enumerate(chunks):
                self.status.emit(f"Summarizing part {i + 1}/{len(chunks)}...")
                word_count = len(chunk.split())
                chunk_max = max_len if word_count > max_len else max(20, word_count // 2)
                chunk_min = min(min_len, max(10, chunk_max - 10))
                summary = self._summarize_chunk(model, tokenizer, chunk, chunk_min, chunk_max)
                summaries.append(summary)

            final_summary = " ".join(summaries)
            self.finished.emit(final_summary)
        except Exception as exc:
            self.error.emit(str(exc))

    @staticmethod
    def _chunk_text(text, max_chars=3000):
        text = text.strip()
        if len(text) <= max_chars:
            return [text]
        # Split on paragraph/sentence boundaries so we don't cut mid-sentence.
        chunks, current = [], ""
        for sentence in text.replace("\n", " ").split(". "):
            sentence = sentence.strip()
            if not sentence:
                continue
            candidate = f"{current}. {sentence}" if current else sentence
            if len(candidate) > max_chars and current:
                chunks.append(current + ".")
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks


# --------------------------------------------------------------------------
# Main Window
# --------------------------------------------------------------------------
class TextSummarizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self._build_ui()

    # ---------------------------- UI BUILD -----------------------------
    def _build_ui(self):
        self.setWindowTitle("Text Summarizer  •  AI Reading Assistant")
        self.resize(980, 720)
        self.setMinimumSize(760, 560)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 24, 28, 24)
        root.setSpacing(16)

        # ---------- Header ----------
        header = QVBoxLayout()
        header.setSpacing(2)

        title = QLabel("📝  Text Summarizer")
        title.setObjectName("title")
        subtitle = QLabel("Paste a long article below and get an instant AI-generated summary")
        subtitle.setObjectName("subtitle")

        header.addWidget(title)
        header.addWidget(subtitle)
        root.addLayout(header)

        # ---------- Content card ----------
        card = QFrame()
        card.setObjectName("card")
        card_shadow = QGraphicsDropShadowEffect()
        card_shadow.setBlurRadius(40)
        card_shadow.setOffset(0, 8)
        card_shadow.setColor(QColor(0, 0, 0, 160))
        card.setGraphicsEffect(card_shadow)

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(14)

        # Input label + counter row
        input_row = QHBoxLayout()
        input_label = QLabel("ORIGINAL ARTICLE")
        input_label.setObjectName("sectionLabel")
        self.word_count_label = QLabel("0 words")
        self.word_count_label.setObjectName("counterLabel")
        input_row.addWidget(input_label)
        input_row.addStretch()
        input_row.addWidget(self.word_count_label)
        card_layout.addLayout(input_row)

        self.input_text = QTextEdit()
        self.input_text.setObjectName("inputBox")
        self.input_text.setPlaceholderText(
            "Paste your article, blog post, or any long text here..."
        )
        self.input_text.textChanged.connect(self._update_word_count)
        self.input_text.setMinimumHeight(180)
        card_layout.addWidget(self.input_text)

        # Controls row
        controls_row = QHBoxLayout()
        controls_row.setSpacing(12)

        length_label = QLabel("Summary length:")
        length_label.setObjectName("smallLabel")
        self.length_combo = QComboBox()
        self.length_combo.addItems(["Short", "Medium", "Long"])
        self.length_combo.setCurrentText("Medium")
        self.length_combo.setObjectName("combo")

        self.summarize_btn = QPushButton("✨  Summarize")
        self.summarize_btn.setObjectName("primaryBtn")
        self.summarize_btn.setCursor(Qt.PointingHandCursor)
        self.summarize_btn.clicked.connect(self.handle_summarize)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("secondaryBtn")
        self.clear_btn.setCursor(Qt.PointingHandCursor)
        self.clear_btn.clicked.connect(self.handle_clear)

        controls_row.addWidget(length_label)
        controls_row.addWidget(self.length_combo)
        controls_row.addStretch()
        controls_row.addWidget(self.clear_btn)
        controls_row.addWidget(self.summarize_btn)
        card_layout.addLayout(controls_row)

        # Status label
        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        card_layout.addWidget(self.status_label)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")
        card_layout.addWidget(divider)

        # Output label + copy button
        output_row = QHBoxLayout()
        output_label = QLabel("SUMMARY")
        output_label.setObjectName("sectionLabel")
        self.copy_btn = QPushButton("📋 Copy")
        self.copy_btn.setObjectName("copyBtn")
        self.copy_btn.setCursor(Qt.PointingHandCursor)
        self.copy_btn.clicked.connect(self.handle_copy)
        output_row.addWidget(output_label)
        output_row.addStretch()
        output_row.addWidget(self.copy_btn)
        card_layout.addLayout(output_row)

        self.output_text = QTextEdit()
        self.output_text.setObjectName("outputBox")
        self.output_text.setReadOnly(True)
        self.output_text.setPlaceholderText("Your summary will appear here...")
        self.output_text.setMinimumHeight(140)
        card_layout.addWidget(self.output_text)

        root.addWidget(card)

        footer = QLabel("Powered by PyQt5 + HuggingFace Transformers (DistilBART)")
        footer.setObjectName("footer")
        footer.setAlignment(Qt.AlignCenter)
        root.addWidget(footer)

        self.setStyleSheet(STYLE_SHEET)

    # ---------------------------- BEHAVIOR -----------------------------
    def _update_word_count(self):
        text = self.input_text.toPlainText().strip()
        count = len(text.split()) if text else 0
        self.word_count_label.setText(f"{count} words")

    def handle_clear(self):
        self.input_text.clear()
        self.output_text.clear()
        self.status_label.setText("")

    def handle_copy(self):
        summary = self.output_text.toPlainText().strip()
        if not summary:
            return
        QApplication.clipboard().setText(summary)
        self.status_label.setText("✅ Summary copied to clipboard!")

    def handle_summarize(self):
        text = self.input_text.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "No text", "Please paste some article text first.")
            return
        if len(text.split()) < 25:
            QMessageBox.warning(
                self, "Text too short",
                "Please paste a longer article (at least ~25 words) for a meaningful summary."
            )
            return

        self.summarize_btn.setEnabled(False)
        self.summarize_btn.setText("Working...")
        self.output_text.clear()
        self.status_label.setText("Preparing...")

        length_mode = self.length_combo.currentText()
        self.worker = SummarizerWorker(text, length_mode)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self._on_summary_ready)
        self.worker.error.connect(self._on_summary_error)
        self.worker.start()

    def _on_summary_ready(self, summary):
        self.output_text.setPlainText(summary)
        self.status_label.setText("✅ Done!")
        self.summarize_btn.setEnabled(True)
        self.summarize_btn.setText("✨  Summarize")

    def _on_summary_error(self, message):
        self.status_label.setText("❌ Something went wrong.")
        self.summarize_btn.setEnabled(True)
        self.summarize_btn.setText("✨  Summarize")
        QMessageBox.critical(
            self, "Error",
            f"Could not generate a summary.\n\nDetails: {message}\n\n"
            "Tip: make sure you're connected to the internet the first time "
            "you run this (the AI model needs to download once)."
        )


# --------------------------------------------------------------------------
# Stylesheet — dark, modern gradient theme
# --------------------------------------------------------------------------
STYLE_SHEET = """
QWidget {
    background-color: #0f0c29;
    font-family: 'Segoe UI', 'Poppins', Arial, sans-serif;
}

QWidget#TextSummarizerApp {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0f0c29, stop:0.5 #302b63, stop:1 #24243e);
}

QLabel#title {
    color: #ffffff;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: 0.5px;
}

QLabel#subtitle {
    color: #b6b0d9;
    font-size: 13px;
    font-weight: 400;
}

QFrame#card {
    background-color: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.10);
    border-radius: 18px;
}

QLabel#sectionLabel {
    color: #9f8fff;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

QLabel#smallLabel {
    color: #cfc9ee;
    font-size: 12px;
}

QLabel#counterLabel {
    color: #7d76a8;
    font-size: 11px;
}

QLabel#statusLabel {
    color: #4ade80;
    font-size: 12px;
    font-weight: 600;
    padding-top: 2px;
}

QLabel#footer {
    color: #5f5a86;
    font-size: 11px;
    padding-top: 6px;
}

QFrame#divider {
    background-color: rgba(255, 255, 255, 0.10);
    max-height: 1px;
    border: none;
}

QTextEdit#inputBox, QTextEdit#outputBox {
    background-color: rgba(15, 12, 41, 0.55);
    border: 1.5px solid rgba(159, 143, 255, 0.25);
    border-radius: 12px;
    padding: 12px;
    color: #f1eefc;
    font-size: 14px;
    selection-background-color: #7c5cff;
}

QTextEdit#inputBox:focus, QTextEdit#outputBox:focus {
    border: 1.5px solid #9f8fff;
}

QTextEdit#outputBox {
    background-color: rgba(124, 92, 255, 0.10);
    border: 1.5px solid rgba(124, 92, 255, 0.35);
}

QComboBox#combo {
    background-color: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(159, 143, 255, 0.35);
    border-radius: 8px;
    padding: 6px 12px;
    color: #f1eefc;
    min-width: 90px;
}

QComboBox#combo::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #24243e;
    color: #f1eefc;
    selection-background-color: #7c5cff;
    border: 1px solid rgba(159, 143, 255, 0.35);
}

QPushButton#primaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #7c5cff, stop:1 #ff5ca0);
    color: white;
    font-weight: 700;
    font-size: 14px;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
}

QPushButton#primaryBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #8d70ff, stop:1 #ff77b3);
}

QPushButton#primaryBtn:disabled {
    background: #4b4470;
    color: #a89fd6;
}

QPushButton#secondaryBtn, QPushButton#copyBtn {
    background-color: rgba(255, 255, 255, 0.08);
    color: #e2defc;
    font-size: 13px;
    font-weight: 600;
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 10px;
    padding: 10px 18px;
}

QPushButton#secondaryBtn:hover, QPushButton#copyBtn:hover {
    background-color: rgba(255, 255, 255, 0.16);
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: rgba(159, 143, 255, 0.45);
    border-radius: 4px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def main():
    app = QApplication(sys.argv)
    window = TextSummarizerApp()
    window.setObjectName("TextSummarizerApp")
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()