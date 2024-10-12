import argparse
import os
import traceback
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    log_mel_spectrogram,
    pad_or_trim,
)
from .decoding import DecodingOptions, DecodingResult, CustomDecodingResult # CODE CHANGE
from .timing import add_word_timestamps
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import (
    exact_div,
    format_timestamp,
    get_end,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)

if TYPE_CHECKING:
    from .model import Whisper

def custom_audio_embedding(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor]
):
    """
    Get the embeddings of an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance
    
    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform


    Returns
    -------
    Audio Embeddings
    """

    """
    Convert audio into log-mel spectrogram, an input format suitable for Whisper
    """
    mel = log_mel_spectrogram(audio, model.dims.n_mels)
    return model.embed_audio(mel)