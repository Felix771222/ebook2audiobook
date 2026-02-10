from lib.conf_models import TTS_ENGINES, default_engine_settings

models = {
    "Qwen3-TTS-12Hz-1.7B-CustomVoice": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS",
        "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "type": "custom_voice",
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3']]['samplerate']
    },
    "Qwen3-TTS-12Hz-1.7B-VoiceDesign": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS",
        "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        "type": "voice_design",
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3']]['samplerate']
    },
    "Qwen3-TTS-12Hz-1.7B-Base": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS",
        "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        "type": "voice_clone",
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3']]['samplerate']
    },
    "Qwen3-TTS-12Hz-0.6B-CustomVoice": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS",
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "type": "custom_voice",
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3']]['samplerate']
    },
    "Qwen3-TTS-12Hz-0.6B-Base": {
        "lang": "multi",
        "repo": "Qwen/Qwen3-TTS",
        "model_name": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "type": "voice_clone",
        "samplerate": default_engine_settings[TTS_ENGINES['QWEN3']]['samplerate']
    }
}
