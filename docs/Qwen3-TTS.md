# Qwen3-TTS Integration for ebook2audiobook

This document describes the integration of Qwen3-TTS into ebook2audiobook.

## Installation

Before using Qwen3-TTS, you need to install the required dependencies:

```bash
pip install qwen-tts
```

Or install all dependencies from requirements.txt:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# GUI mode
ebook2audiobook.command --tts_engine QWEN3

# Headless mode
ebook2audiobook.command --headless --ebook /path/to/ebook --tts_engine QWEN3 --language eng
```

### Available Models

| Model | Type | Description |
|-------|------|-------------|
| Qwen3-TTS-12Hz-1.7B-CustomVoice | Custom Voice | 9 premium timbres with style control |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | Voice Design | Natural language voice design |
| Qwen3-TTS-12Hz-1.7B-Base | Voice Clone | 3-second rapid voice clone |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | Custom Voice | Lightweight version |
| Qwen3-TTS-12Hz-0.6B-Base | Voice Clone | Lightweight version |

### Supported Languages

- Chinese (zho)
- English (eng)
- Japanese (jpn)
- Korean (kor)
- German (deu)
- French (fra)
- Russian (rus)
- Portuguese (por)
- Spanish (spa)
- Italian (ita)

### Voice Options

For CustomVoice models, available speakers include:

| Speaker | Description | Native Language |
|---------|-------------|----------------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male voice | Chinese |
| Dylan | Youthful Beijing male | Chinese (Beijing Dialect) |
| Eric | Lively Chengdu male | Chinese (Sichuan Dialect) |
| Ryan | Dynamic male voice | English |
| Aiden | Sunny American male | English |
| Ono_Anna | Playful Japanese female | Japanese |
| Sohee | Warm Korean female | Korean |

### Voice Cloning

To use voice cloning with Qwen3-TTS Base models:

```bash
ebook2audiobook.command --headless --ebook /path/to/ebook --tts_engine QWEN3 \
  --fine_tuned Qwen3-TTS-12Hz-1.7B-Base \
  --voice /path/to/reference_audio.wav
```

### GPU Requirements

- **VRAM**: 4GB minimum (recommended 8GB+)
- **RAM**: 4GB minimum
- **Device**: CUDA, CPU, MPS, ROCm, XPU, JETSON

## Features

- **High Quality Speech**: State-of-the-art speech synthesis quality
- **Voice Cloning**: Rapid 3-second voice cloning from reference audio
- **Voice Design**: Natural language-based voice design
- **Multi-language**: Supports 10 major languages
- **Streaming**: Low-latency streaming generation (~97ms first audio packet)
- **Instruction Control**: Control tone, speaking rate, and emotional expression

## Performance Ratings

| Metric | Rating |
|--------|--------|
| VRAM Usage | ⭐⭐⭐⭐ (4GB) |
| CPU Usage | ⭐⭐ (2) |
| RAM Usage | ⭐⭐⭐⭐ (4GB) |
| Realism | ⭐⭐⭐⭐⭐ (5) |

## Model Download

Models are automatically downloaded from Hugging Face when first used:

- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`
- `Qwen/Qwen3-TTS-12Hz-0.6B-Base`

For China users, models can also be downloaded from ModelScope.

## Troubleshooting

### Installation Issues

If you encounter installation errors:

```bash
# Create a fresh environment
conda create -n ebook2audiobook-qwen3 python=3.12 -y
conda activate ebook2audiobook-qwen3
pip install qwen-tts

# If using GPU, install FlashAttention
pip install -U flash-attn --no-build-isolation
```

### GPU Memory Issues

If you experience VRAM issues:

1. Reduce batch size
2. Use CPU instead of GPU for large files
3. Set `device: CPU` in configuration

### Model Loading Failures

If models fail to load:

1. Check internet connection (for Hugging Face downloads)
2. Verify Hugging Face credentials if using private models
3. Check disk space in cache directory

## Credits

- **Qwen Team at Alibaba Cloud** for developing Qwen3-TTS
- **Qwen3-TTS Repository**: https://github.com/QwenLM/Qwen3-TTS
- **Hugging Face**: https://huggingface.co/collections/Qwen/qwen3-tts
