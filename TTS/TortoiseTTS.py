import random
import requests
from requests.exceptions import JSONDecodeError
from utils import settings

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HF hub.
tts = TextToSpeech()

voices = [
    "angie",
    "applejack",
    "cond_latent_example",
    "daniel",
    "deniro",
    "emma",
    "freeman",
    "geralt",
    "halle",
    "jlaw",
    "lj",
    "mol",
    "pat",
    "pat2",
    "rainbow",
    "snakes",
    "tim_reynolds",
    "tom",
    "train_atkins",
    "train_daws",
    "train_dotrice",
    "train_dreams",
    "train_empire",
    "train_grace",
    "train_kennard",
    "train_lescault",
    "train_mouse",
    "weaver",
    "william",
]


# valid voices https://lazypy.ro/tts/


class TortoiseTTS:
    def __init__(self):
        self.url = "https://streamlabs.com/polly/speak"
        self.max_chars = 350
        self.voices = voices

    def run(self, text, filepath, random_voice: bool = False):
        if random_voice:
            voice = self.randomvoice()
        else:
            if not settings.config["settings"]["tts"]["tortoise_voice"]:
                raise ValueError(
                    f"Please set the config variable TORTOISE to a valid voice. options are: {voices}"
                )
            voice = str(settings.config["settings"]["tts"]
                        ["tortoise_voice"])

        # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
        # preset = "standard"
        if settings.config["settings"]["tts"]["tortoise_preset"]:
            preset = settings.config["settings"]["tts"]["tortoise_preset"]
        else:
            preset = "fast"

        voice_samples, conditioning_latents = load_voice(voice)
        gen = tts.tts_with_preset(text, voice_samples=voice_samples, conditioning_latents=conditioning_latents,
                                  preset=preset)
        torchaudio.save(filepath, gen.squeeze(0).cpu(), 24000)

    def randomvoice(self):
        return random.choice(self.voices)
