from lib.classes.tts_engines.common.headers import *
from lib.classes.tts_engines.common.preset_loader import load_engine_presets
from lib.conf_models import default_engine_settings
import torch


class Qwen3TTS(TTSUtils, TTSRegistry, name='qwen3'):

    def __init__(self, session: DictProxy):
        try:
            self.session = session
            self.cache_dir = tts_dir
            self.speakers_path = None
            self.speaker = None
            self.tts_key = self.session['model_cache']
            self.pth_voice_file = None
            self.resampler_cache = {}
            self.audio_segments = []
            self.models = load_engine_presets(self.session['tts_engine'])
            self.params = {}
            self.params['samplerate'] = self.models[self.session['fine_tuned']]['samplerate']
            self.languages = default_engine_settings[TTS_ENGINES['QWEN3']]['languages']
            enough_vram = self.session['free_vram_gb'] > 4.0
            seed = 0
            self.amp_dtype = self._apply_gpu_policy(enough_vram=enough_vram, seed=seed)
            self.engine = self.load_engine()
        except Exception as e:
            error = f'__init__() error: {e}'
            raise ValueError(error)

    def load_engine(self) -> Any:
        try:
            try:
                from qwen_tts import Qwen3TTSModel
                import torch
            except ImportError as e:
                raise ImportError(f"qwen-tts package not installed. Run: pip install qwen-tts. Error: {e}")

            msg = f'Loading TTS {self.tts_key} model, it takes a while, please be patient…'
            print(msg)
            self._cleanup_memory()

            engine = loaded_tts.get(self.tts_key)
            if not engine:
                try:
                    model_name = self.models[self.session['fine_tuned']]['model_name']

                    dtype = torch.bfloat16 if self.session['device'] in ['cuda', 'GPU'] else torch.float32

                    device_map = "cuda:0" if self.session['device'] in ['cuda', 'GPU'] else "cpu"

                    engine = Qwen3TTSModel.from_pretrained(
                        model_name,
                        device_map=device_map,
                        dtype=dtype,
                    )
                    loaded_tts[self.tts_key] = engine
                except Exception as e:
                    error = f'load_engine(): Qwen3-TTS model loading failed: {e}'
                    raise RuntimeError(error) from e

            if engine:
                msg = f'TTS {self.tts_key} Loaded!'
                print(msg)
                return engine
            error = 'load_engine(): engine is None'
            raise RuntimeError(error)
        except Exception as e:
            error = f'load_engine() error: {e}'
            raise RuntimeError(error)

    def convert(self, sentence_number: int, sentence: str) -> bool:
        try:
            import numpy as np
            import torch
            import torchaudio
            from lib.classes.tts_engines.common.audio import trim_audio

            if self.engine:
                final_sentence_file = os.path.join(self.session['sentences_dir'], f'{sentence_number}.{default_audio_proc_format}')
                device = devices['CUDA']['proc'] if self.session['device'] in [devices['CUDA']['proc'], devices['JETSON']['proc']] else self.session['device']

                sentence_parts = self._split_sentence_on_sml(sentence)
                self.audio_segments = []

                lang_code = self.languages.get(self.session['language'], 'English')

                for part in sentence_parts:
                    part = part.strip()
                    if not part:
                        continue

                    if self._is_sml_tag(part):
                        converted = self._convert_sml(part)
                        if converted is not None:
                            self.audio_segments.append(trim_audio(converted, 30))
                        continue

                    msg = f'Converting to speech: {part[:50]}...'
                    print(msg)

                    try:
                        if self._is_voice_clone_mode():
                            wavs, sr = self._generate_voice_clone(part, lang_code)
                        else:
                            wavs, sr = self._generate_custom_voice(part, lang_code)

                        if wavs is not None and len(wavs) > 0:
                            audio_data = wavs[0]
                            if isinstance(audio_data, np.ndarray):
                                audio_data = torch.from_numpy(audio_data)
                            elif not isinstance(audio_data, torch.Tensor):
                                audio_data = torch.tensor(audio_data)

                            if sr != self.params['samplerate']:
                                audio_data = torchaudio.functional.resample(audio_data, sr, self.params['samplerate'])

                            audio_data = trim_audio(audio_data, 30)
                            self.audio_segments.append(audio_data)
                    except Exception as e:
                        msg = f'Error converting text to speech: {e}'
                        print(msg)
                        continue

                if self.audio_segments:
                    combined_audio = torch.cat(self.audio_segments, dim=0)
                    import soundfile as sf
                    sf.write(final_sentence_file, combined_audio.cpu().numpy(), self.params['samplerate'])
                    msg = f'Audio saved: {final_sentence_file}'
                    print(msg)
                    return True
                else:
                    msg = 'No audio generated'
                    print(msg)
                    return False
            else:
                msg = 'Engine not loaded'
                print(msg)
                return False
        except Exception as e:
            error = f'convert() error: {e}'
            raise RuntimeError(error)

    def _is_voice_clone_mode(self) -> bool:
        return self.session.get('voice_clone', False) and self.params.get('current_voice') is not None

    def _generate_custom_voice(self, text: str, language: str) -> tuple:
        speaker = self._get_default_speaker()
        instruct = self.session.get('tts_instruct', '')

        wavs, sr = self.engine.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct else None,
        )
        return wavs, sr

    def _generate_voice_clone(self, text: str, language: str) -> tuple:
        ref_audio = self.params.get('current_voice')
        ref_text = self.session.get('ref_text', '')

        if ref_audio is None:
            raise ValueError("Voice clone requires reference audio")

        voice_clone_prompt = None
        if hasattr(self, '_voice_clone_prompt') and self._voice_clone_prompt is not None:
            voice_clone_prompt = self._voice_clone_prompt

        wavs, sr = self.engine.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text if ref_text else None,
            voice_clone_prompt=voice_clone_prompt,
        )
        return wavs, sr

    def _get_default_speaker(self) -> str:
        model_config = self.models[self.session['fine_tuned']]
        if 'speakers' in model_config and model_config['speakers']:
            return list(model_config['speakers'].keys())[0]
        return 'Vivian'

    def _is_sml_tag(self, text: str) -> bool:
        return text.strip().startswith('[') and text.strip().endswith(']')

    def _convert_sml(self, sml: str) -> torch.Tensor | None:
        from lib.classes.tts_engines.common.audio import trim_audio
        import re

        sml = sml.strip()[1:-1]
        match = re.match(r'(\w+)(?::(.+))?', sml)
        if not match:
            return None

        tag_name = match.group(1)
        tag_value = match.group(2)

        if tag_name == 'break':
            import random
            duration = random.uniform(0.3, 0.6) if not tag_value else float(tag_value)
            sr = self.params['samplerate']
            silence = torch.zeros(int(sr * duration))
            return trim_audio(silence, 30)

        elif tag_name == 'pause':
            import random
            duration = random.uniform(1.0, 1.6) if not tag_value else float(tag_value)
            sr = self.params['samplerate']
            silence = torch.zeros(int(sr * duration))
            return trim_audio(silence, 30)

        elif tag_name == 'voice' and tag_value:
            voice_path = tag_value.strip()
            self.params['current_voice'] = voice_path
            self._voice_clone_prompt = None
            return None

        return None

    def set_voice(self, voice: str):
        self.speaker = voice
        self.params['current_voice'] = voice

    def get_supported_speakers(self) -> list:
        model_config = self.models[self.session['fine_tuned']]
        return list(model_config.get('speakers', {}).keys())

    def get_supported_languages(self) -> dict:
        return self.models[self.session['fine_tuned']].get('languages', {})

    def _cleanup_memory(self):
        if self.session['device'] in ['cuda', 'GPU', 'JETSON']:
            import torch
            torch.cuda.empty_cache()

    def create_vtt(self, all_sentences: list) -> bool:
        audio_dir = self.session['sentences_dir']
        vtt_path = os.path.join(self.session['process_dir'], Path(self.session['final_name']).stem + '.vtt')
        if self._build_vtt_file(all_sentences, audio_dir, vtt_path):
            return True
        return False
