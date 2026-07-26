# ✨ Text Summarizer - Professional Edition

**A powerful AI-driven text summarization application with a beautiful GUI**

## 🎯 Problem Solved
Readers lack time to go through long articles. This application instantly summarizes lengthy texts, helping you save time and extract key information quickly.

## ⚡ Features
- **Fast Summarization**: Uses advanced BART model from Facebook/Meta
- **Beautiful GUI**: Modern dark theme with gradient buttons and smooth animations
- **Customizable Output**: Adjust min/max length of summaries
- **Real-time Progress**: Visual feedback during summarization
- **Copy to Clipboard**: Easy sharing of results
- **Text Statistics**: Shows compression ratio and word counts
- **Threading**: Non-blocking UI - app stays responsive while processing

## 🛠️ Technology Stack
- **PyQt5**: Modern GUI framework
- **Transformers**: State-of-the-art NLP models from Hugging Face
- **BART Model**: Facebook's large conditional generation model
- **Python 3.8+**: Backend logic

## 📥 Installation (EASY!)

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

That's it! No virtual environment needed.

## 🚀 Running the Application

### Option 1: Direct Python
```bash
python summarizer.py
```

### Option 2: Create a Batch File (Windows)
Create a file named `run.bat` in the same folder:
```batch
@echo off
python summarizer.py
pause
```
Then double-click `run.bat` to run the app.

### Option 3: Command Line
```bash
cd "d:\Text Summarizer"
python summarizer.py
```

## 💡 How to Use

1. **Launch the Application**: Run the script using one of the methods above
2. **Paste Your Text**: Copy and paste an article or long text into the input area
3. **Adjust Settings** (Optional):
   - **Max Length**: Maximum words in summary (default: 150)
   - **Min Length**: Minimum words in summary (default: 50)
4. **Click "⚡ SUMMARIZE NOW"**: Wait for the AI to process
5. **View Results**: Your summary appears in the output area
6. **Copy Result**: Click "📋 COPY RESULT" to copy to clipboard
7. **Clear**: Use "🗑️ CLEAR ALL" to reset

## 🎨 Design Features
- **Dark Modern Theme**: Easy on the eyes, professional look
- **Gradient Buttons**: Cyan (summarize), Green (copy), Red (clear)
- **Responsive UI**: Doesn't freeze during processing
- **Progress Indicators**: Real-time feedback on what's happening
- **Statistics Panel**: Shows compression metrics

## 📊 Example

**Input** (50 words):
> Artificial intelligence is transforming industries. It helps doctors diagnose diseases better. AI powers self-driving cars. Businesses use AI for customer service. The technology continues to evolve rapidly...

**Output** (15 words):
> Artificial intelligence transforms industries including healthcare, transportation, and business customer service.

**Compression**: 70% reduction!

## 🔧 Troubleshooting

### "ModuleNotFoundError" Error
**Solution**: Run `pip install -r requirements.txt` again

### Application Takes Time to Start
**Reason**: First run downloads the BART model (~1.6GB). Subsequent runs are instant.

### Text Not Summarizing
**Solution**: Make sure you have at least 30 words of text

### Out of Memory Error
**Solution**: Try summarizing shorter texts, or reduce Max Length value

## 📝 Customization Tips

Edit `summarizer.py` to:
- Change colors: Look for `#00D4FF`, `#00C86B`, `#FF6B6B`
- Change font: Look for `"Segoe UI"` 
- Change model: Replace `"facebook/bart-large-cnn"` with other models
- Adjust sizes: Look for `setMinimumHeight()`, `setGeometry()`

## 🤝 Model Information

Uses **facebook/bart-large-cnn** by Meta:
- Trained on CNN/DailyMail dataset
- Optimized for news article summarization
- Fast and accurate
- ~1.6GB download (one-time)

## 📚 Supported Input
- News articles
- Blog posts
- Research papers (shorter sections)
- Technical documentation
- Any long-form text (minimum 50 words recommended)

## 🚫 Limitations
- First run takes 2-3 minutes (model download)
- Requires stable internet for first run
- Best results with 100+ words input
- Works best with English text

## 💻 System Requirements
- Python 3.8 or higher
- 2GB RAM minimum (4GB recommended)
- 2GB disk space (for model)
- Windows/Mac/Linux

## 📄 License
Free to use and modify!

---

**Made with ❤️ for busy readers everywhere!**

Enjoy your Text Summarizer! 🎉
