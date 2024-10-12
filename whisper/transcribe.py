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


def transcribe(
    model: "Whisper",
    audio: Union[str, np.ndarray, torch.Tensor],
    *,
    verbose: Optional[bool] = None,
    temperature: Union[float, Tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    condition_on_previous_text: bool = True,
    initial_prompt: Optional[str] = None,
    word_timestamps: bool = False,
    prepend_punctuations: str = "\"'“¿([{-",
    append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
    clip_timestamps: Union[str, List[float]] = "0",
    hallucination_silence_threshold: Optional[float] = None,
    **decode_options,
):
    """
    Transcribe an audio file using Whisper

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    audio: Union[str, np.ndarray, torch.Tensor]
        The path to the audio file to open, or the audio waveform

    verbose: bool
        Whether to display the text being decoded to the console. If True, displays all the details,
        If False, displays minimal details. If None, does not display anything

    temperature: Union[float, Tuple[float, ...]]
        Temperature for sampling. It can be a tuple of temperatures, which will be successively used
        upon failures according to either `compression_ratio_threshold` or `logprob_threshold`.

    compression_ratio_threshold: float
        If the gzip compression ratio is above this value, treat as failed

    logprob_threshold: float
        If the average log probability over sampled tokens is below this value, treat as failed

    no_speech_threshold: float
        If the no_speech probability is higher than this value AND the average log probability
        over sampled tokens is below `logprob_threshold`, consider the segment as silent

    condition_on_previous_text: bool
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.

    word_timestamps: bool
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

    prepend_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the next word

    append_punctuations: str
        If word_timestamps is True, merge these punctuation symbols with the previous word

    initial_prompt: Optional[str]
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.

    decode_options: dict
        Keyword arguments to construct `DecodingOptions` instances

    clip_timestamps: Union[str, List[float]]
        Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process.
        The last end timestamp defaults to the end of the file.

    hallucination_silence_threshold: Optional[float]
        When word_timestamps is True, skip silent periods longer than this threshold (in seconds)
        when a possible hallucination is detected

    Returns
    -------
    A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
    the spoken language ("language"), which is detected when `decode_options["language"]` is None.
    """

    """
    Data type setup 
    """
    dtype = torch.float16 if decode_options.get("fp16", True) else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn("Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn("FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32

    if dtype == torch.float32:
        decode_options["fp16"] = False





    """
    Convert audio into log-mel spectrogram, an input format suitable for Whisper
    content_frames: Number of frames in the log-mel spectrogram that contain actual audio
    content_duration: Total duration of actual audio content
    """
    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio, model.dims.n_mels, padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_FRAMES
    content_duration = float(content_frames * HOP_LENGTH / SAMPLE_RATE)





    """
    Detect the language
    """
    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up to the first 30 seconds. Use `--language` to specify the language"
                )
            mel_segment = pad_or_trim(mel, N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options['language']].title()}"
                )



    """
    Initialise the language, model task and tokenizer
    """
    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language=language,
        task=task,
    )




    """
    clip_timestamps input by user is a Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process
    [start transcribe time, end transcribe time, start transcribe time, end transcribe time]

    seek_clips, the end result is a list of clips to transcribe in the format
    [(start frame index,end frame index),(start frame index,end frame index)]

    


    IF NO clip_timestamps IS GIVEN (PROBABLY OUR CASE),
    seek_clips results in
    [(0, content_frames)] i.e. [(0, number of frames that actually contain audio)]

    so in the while loop below, it only iterates once
    """
    if isinstance(clip_timestamps, str):
        clip_timestamps = [
            float(ts) for ts in (clip_timestamps.split(",") if clip_timestamps else [])
        ]
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND) for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2], seek_points[1::2]))

    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "translate":
        warnings.warn("Word-level timestamps on translations may not be reliable.")







    """
    Decode an audio segment into text. Returns a DecodingResult type

    Initialise the temperatures to use: Controls the randomness of predictions
    Low temp: More deterministic. High temp: More randomness

    For each temperature, attempt decoding with each one
    - Get the decoding options
    - Decode with model.decode and the options that returns DecodingResult
    - Evaluate the decoding quality and decide if we should continue on to the next temp
    
    Return the decoding result
    - CustomDecodingResult(
                audio_features=audio_features,
                language=languages,
                tokens=tokens,
                texts=texts,
                avg_logprob=avg_logprobs[0],
                no_speech_prob=no_speech_probs[0],
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(texts[0]))
    """
    def decode_with_fallback(segment: torch.Tensor) -> CustomDecodingResult: #CODE CHANGE
        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = DecodingOptions(**kwargs, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and decode_result.compression_ratio > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return decode_result





    """
    clip_idx: Keep track of which clip in seek_clips we are in

    Recall seek_clips = [(start frame index,end frame index),(start frame index,end frame index)]
    or [(0, content_frames)]

    seek: Set the initial frame number of the clip we are processing i.e. start frame index

    input_stride: Determine the number of mel spectogram frames that correspond to 1 output token
    ratio of number of mel frames to output tokens

    time_precision: Calculate the duration in seconds that corresponds to
    one output token

    all_tokens: Store all the tokens generated during transcription
    - accumulate the token IDs corresponding to the transcribed text
    - used to reconstruct the fully transcribed text at the end

    all_segments: Store the dictionaries representing each segment of the transcription

    prompt_reset_since: 
    """
    # clip_idx = 0 # CODE CHANGE: NOT USING MULTIPLE CLIPS
    # seek = seek_clips[clip_idx][0] # CODE CHANGE: NOT USING MULTIPLE CLIPS
    seek = 0 # Start from the first frame index which is 0 since we are starting from the very start, no cases where we are starting from the second onwards clip
    input_stride = exact_div(
        N_FRAMES, model.dims.n_audio_ctx
    )  # mel frames per output token: 2
    time_precision = (
        input_stride * HOP_LENGTH / SAMPLE_RATE
    )  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0





    """
    For prompt engineering
    """
    if initial_prompt is not None:
        initial_prompt_tokens = tokenizer.encode(" " + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []





    """
    Returns a dictionary representing a transcription segment

    This dictionary segment is added to the all_segments list

    Accepts the 
    - start, end time of the segment
    - token IDs for this segment
    - DecodingResult

    We filter out the tokens that have IDs less than eot i.e. only want transcribed text
    - tokens greater than tokenizer.eot are special tokens
    """
    def new_segment(
        *, start: float, end: float, tokens: torch.Tensor, result: CustomDecodingResult # CODE CHANGE
    ):
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens if token < tokenizer.eot]
        return {
            "seek": seek,
            "start": start,
            "end": end,
            "text": tokenizer.decode(text_tokens),
            "tokens": tokens,
            "temperature": result.temperature,
            "avg_logprob": result.avg_logprob,
            "compression_ratio": result.compression_ratio,
            "no_speech_prob": result.no_speech_prob,
        }





    """
    Set up the progress bar to display the transcription progress 
    The total number of iterations for the progress bar corresponds to 
    the total number of content frames in the audio
    """

    
    # show the progress bar when verbose is False (if True, transcribed text will be printed)
    """
    CODE CHANGE: Dont need progress bar
    with tqdm.tqdm(total=content_frames, unit="frames", disable=verbose is not False) as pbar:
    """ 
    # CODE CHANGE: Dont need this value because we are not doing word level timestamps
    # last_speech_timestamp = 0.0
    # NOTE: This loop is obscurely flattened to make the diff readable.
    # A later commit should turn this into a simpler nested loop.
    # for seek_clip_start, seek_clip_end in seek_clips:
    #     while seek < seek_clip_end





    """
    PROCESSING CLIPS: EACH CLIP OR THE ENTIRE AUDIO IF NO CLIP SPECIFICATION

    Set up a loop to iterate over the audio clips in seek_clips
    recall: seek_clips
    [(start frame index,end frame index),(start frame index,end frame index)]
    [(0, content_frames)]

    Recall that clip_idx is the index of the current clip in seek_clips we are in,
    initialised to 0

    Unpack the start frame and end frame of the current clip

    Adjust the seek position. seek holds the current frame number

    if seek is before the start of the current clip, move it to the clip's start

    if seek is more than the end of the current clip,
        move to the next clip
        if the next clip is still in seek clips
            Get the start frame index of this next clip
        else it means we have exhausted all clips
    """

    """
    # CODE CHANGE: DONT NEED TO DO A WHILE LOOP BECAUSE WE ARE NOT 
    # GOING TO SPLIT THE AUDIO INTO CLIPS
    while clip_idx < len(seek_clips):
        seek_clip_start, seek_clip_end = seek_clips[clip_idx]

        # Adjust seek 
        if seek < seek_clip_start:
            seek = seek_clip_start
        if seek >= seek_clip_end:
            clip_idx += 1
            if clip_idx < len(seek_clips):
                seek = seek_clips[clip_idx][0]
            continue
    """


    """
    INSTANTIATING SEGMENT INFO (SUBSET OF CLIP)

    time_offset: Calculate the start time of the current segment using seek, the current frame index

    window_end_time: Calculate the end time of the current processing window

    segment_size: Determine the actual number of frames to process in the curr segment

    mel_segment: Extract the portion of the mel-spectrogram corresponding to the curr segment
    - make sure it has the correct size

    segment_duration: Calculate the duration of the curr segment

    window_end_time is the theoretical end time of the processing window,
    using N_FRAMES (num frames per segment) from the current seek position

    segment duration is the actual duration

    Segments are smaller, fixed-size windows within a clip 
    Help to process the audio in manageable chunks
    """
    time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
    # CODE CHANGE: Dont need this because we are not doing word timestamps
    # window_end_time = float((seek + N_FRAMES) * HOP_LENGTH / SAMPLE_RATE)
    # CODE CHANGE: No longer using seek_clip_end because we are not using multiple clips
    # segment_size = min(N_FRAMES, content_frames - seek, seek_clip_end - seek)
    # Let the segment size be determined by N_FRAMES (Desired fixed num frames per segment)
    # or the remaining number of frames in the audio content, whichever is smaller
    segment_size = min(N_FRAMES, content_frames - seek)
    # CODE CHECK: use seek+segment_size as the boundary or seek+N_FRAMES
    mel_segment = mel[:, seek : seek + segment_size]
    segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
    mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

    """
    Perform decoding on the audio segment (mel segment)
    """
    decode_options["prompt"] = all_tokens[prompt_reset_since:]
    
    """
    ORIGINAL CODE
    result: DecodingResult = decode_with_fallback(mel_segment) # calls on a segment (computational requirements)
    tokens = torch.tensor(result.tokens)
    """
    # CODE CHANGE: Change result type returned from decoding.py
    # Decode the current segment, decode the very first mel segment into a transcription
    result: DecodingResult = decode_with_fallback(mel_segment) # calls on a segment (computational requirements)
    # Recall: Seek is the frame index of the clip we are processing
    # CODE CHANGE: Maintain a different seek position for each hypotheses
    # Why add this
    # decode_options["beam_size"]
    # Check if beam size needs to be passed to decoding also
    seeks = [0]*decode_options["beam_size"] # one seek variable for each hypothesis


    """
    First check if we should skip based on the no speech probability and 
    no speech threshold

    Have a second check on the average log probability of the decoded tokens
    to see if the model is confident in its output

    If we relly should skip, increment seek by the size of the segment
    and skip the code below
    """
    if no_speech_threshold is not None:
        # no voice activity check
        should_skip = result.no_speech_prob > no_speech_threshold
        if (
            logprob_threshold is not None
            and result.avg_logprob > logprob_threshold
        ):
            # don't skip if the logprob is high enough, despite the no_speech_prob
            should_skip = False

        if should_skip:
            # CODE CHANGE: We are now maintaining a different seek position for each hypothesis
            # seek += segment_size  # fast-forward to the next segment boundary
            # CODE_CHANGE: no longer in the while loop, so there is no continuing on to a next clip
            # continue
            # Instead, we fast forward to the next segment boundary
            seeks = [seek + segment_size for seek in seeks]  # fast-forward to the next segment boundary

    """
    Store the current seek position
    Initialise a list to store the segments generated in this iteration
    """

    # CODE CHANGE: No longer using previous seek becauuse not using word timestamps and progress bar
    # previous_seek = seek
    # CODE CHANGE: Instantiated in a for loop below - maintain a current segment list
    # for EACH HYPOTHESES
    # current_segments = []

    """
    Calculate a score to indicate how anomalous a word is

    Determines whether a segment is considered anomalous based on the anomaly scores of its words.

    Finds the next segment in a list of segment dictionaries that contains something in the words key
    """
    # anomalous words are very long/short/improbable

    """
    # CODE CHANGE: ONLY USED IN WORD TIME STAMPS
    def word_anomaly_score(word: dict) -> float:
        probability = word.get("probability", 0.0)
        duration = word["end"] - word["start"]
        score = 0.0
        if probability < 0.15:
            score += 1.0
        if duration < 0.133:
            score += (0.133 - duration) * 15
        if duration > 2.0:
            score += duration - 2.0
        return score

    def is_segment_anomaly(segment: Optional[dict]) -> bool:
        if segment is None or not segment["words"]:
            return False
        words = [w for w in segment["words"] if w["word"] not in punctuation]
        words = words[:8]
        score = sum(word_anomaly_score(w) for w in words)
        return score >= 3 or score + 0.01 >= len(words)

    def next_words_segment(segments: List[dict]) -> Optional[dict]:
        return next((s for s in segments if s["words"]), None)
    """


    """
    # CODE CHANGE

    Initialise 
    - current_segments_list: Each index will hold the list of segments for a hypothesis
    - current_tokens_list: Each index will hold the list of tokens for a hypothesis
    """
    current_segments_list = [] # value per each hypothesis, where value is a list of segments, where a segment is a dict
    current_tokens_list = [] # value per each hypothesis, where value is a list


    """
    CODE CHANGE: Instead of processing a single token, we now have multiple token lists/transcriptions.
    We have calculated a mel segment that can be used for all hypotheses as all the
    seek positions start from 0 (common seek position)
    Add a for loop
    """

    # Loop over all hypotheses, only for the first mel segment as they have a common starting seek position
    for j in range(len(result.tokens)):
        """
        timestamp_tokens: Boolean tensor indicating
        which tokens in the token tensor are timestamp tokens
        - Note: .ge is greater than or equal

        single_timestamp_ending: Check if the token sequence ends with 
        a single timestamp token

        consecutive: Identify the indices in tokens tensor
        where two timestamp tokens occur consecutively
        - Add 1 to refer to the second token in each pair of consecutive timestamp tokens
        
        consecutive timestamp tokens: 
        - can indicate silence, pause or non-speech
        - can signal the end of one speech segment and the beginning of another
        """
        # CODE CHANGE
        # - Add a current_segments to contain the current segments for this hypothesis only
        # - And extract the relevant hypotheses
        # Overall segmentation logic remains the same just that hypothesis has replaced token
        current_segments = []
        hypothesis = torch.tensor(result.tokens[j])

        # CODE CHANGE: Get the timestamp tokens from the hypotheses
        # timestamp_tokens: torch.Tensor = tokens.ge(tokenizer.timestamp_begin)
        # Overall logic remains the same just that hypothesis has replaced token
        timestamp_tokens: torch.Tensor = hypothesis.ge(tokenizer.timestamp_begin)
        single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

        consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
        consecutive.add_(1)


        """
        PROCESSING SEGMENTS WITHIN CLIPS AND ADDING THEM TO 
        current_segments

        PROCESSING EACH SEGMENT



        Some tokens are timestamp tokens, used to represent specific points in time in
        the audio

        Each timestamp token represents a time increment

        When the token sequence generated contains consecutive timestamp tokens

        First create a list of indicies that will be used to segment the token 
        sequence at positions where the consecutive timestamp tokens occur. Call it
        slices (slicing the token sequence)

        If the token sequence ends with a single timestamp token, append
        the length of tokens to slices i.e. last index of the 
        token sequence

        Initialise the last_slice to be 0: Used as the starting index
        for slicing the token sequence


        This is the case where there are consecutive timestamp tokens
        Sub-Segments are created based on the positions of the consecutive timestamps
        - recall that segments are subsets of clips

        Segment processing code
        - adds sub segments
        - differs on consecutive or not
        - differs on single timestamp ending or not 
        """
        if len(consecutive) > 0:
            # if the output contains two consecutive timestamp tokens
            slices = consecutive.tolist()
            if single_timestamp_ending:
                # CODE CHANGE: Hypothesis has replaced token. The logic remains the same
                # slices.append(len(tokens))
                slices.append(len(hypothesis))

            last_slice = 0
            """
            For each slice index,
                Extract the tokens between last_slice and current slice

                start_timestamp_pos: Position of the start timestamp token. 
                Integer value indicating how many timestamp steps
                away from the beginning of the timestamp token

                end_timestamp_pos: Position of the end timestamp token. 
                Integer value indicating how many timestamp steps
                away from the beginning of the timestamp token

                use the start_timestamp_pos and end_timestamp_pos
                to calculate the actual times 

                We then create a new segment consisting of the 
                - start time
                - end time
                - tokens in this segment
                - result

            """
            for current_slice in slices:
                # CODE CHANGE: hypothesis replace tokens
                # sliced_tokens = tokens[last_slice:current_slice]
                sliced_tokens = hypothesis[last_slice:current_slice]
                start_timestamp_pos = (
                    sliced_tokens[0].item() - tokenizer.timestamp_begin
                )
                end_timestamp_pos = (
                    sliced_tokens[-1].item() - tokenizer.timestamp_begin
                )
                current_segments.append(
                    new_segment(
                        start=time_offset + start_timestamp_pos * time_precision,
                        end=time_offset + end_timestamp_pos * time_precision,
                        tokens=sliced_tokens,
                        result=result,
                    )
                )
                last_slice = current_slice

            """
            After we have iterated through all the slices,

            if the token sequence ends with a single timestamp token,
            we can move seek (current frame number of the curr clip) to the next segment
            else ...

            MOVE SEEK TO ITS NEXT POSITION
            """
            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                # CODE CHANGE: We are now maintaining a separate seek position
                # for each hypothesis. They may have different timestamp tokens
                # result in different seek increments
                # seek += segment_size
                seeks[j]+=segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                # CODE CHANGE: Hypothesis replace tokens and seek position
                """
                last_timestamp_pos = (
                    tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seek += last_timestamp_pos * input_stride
                """
                last_timestamp_pos = (
                    hypothesis[last_slice - 1].item() - tokenizer.timestamp_begin
                )
                seeks[j] += last_timestamp_pos * input_stride


        # Executed when there are no consecutive timestamp tokens generated
        # by the model during transcription
        else:
            # Initialise the duration to be the segment duration
            # Extract any timestamp tokens present
            duration = segment_duration
            # CODE CHANGE: Replace tokens with hypothesis
            # timestamps = tokens[timestamp_tokens.nonzero().flatten()]
            timestamps = hypothesis[timestamp_tokens.nonzero().flatten()]
            """
            Adjust the segment's end time to match the last timestamp indicated
            by the model, this represents the end time of the segment
            """
            if (
                len(timestamps) > 0
                and timestamps[-1].item() != tokenizer.timestamp_begin
            ):
                # no consecutive timestamps but it has a timestamp; use the last one.
                last_timestamp_pos = (
                    timestamps[-1].item() - tokenizer.timestamp_begin
                )
                duration = last_timestamp_pos * time_precision
            # Append the new segment
            current_segments.append(
                new_segment(
                    start=time_offset,
                    end=time_offset + duration,
                    # Code change: replace tokens wth hypothesis
                    # tokens=tokens,
                    tokens = hypothesis,
                    result=result,
                )
            )
            # Update the seek position to go to the next segment
            # CODE CHANGE: Updated seek position for a hypothesis
            # seek += segment_size
            seeks[j]+=segment_size

        """
        CODE CHANGE: Code addition
        Add the processed current_segments for a hypothesis to its
        correct index in the current_segments_list that contains
        all segments for each hypothesis
        """
        try:
            current_segments_list[j].extend([current_segments])
        except IndexError:
            current_segments_list.append([current_segments])




        """
        Extract word-level timestamps using the cross-attention pattern and dynamic time warping,
        and include the timestamps for each word in each segment.

        By default it is false
        """

        """
        if word_timestamps:
            add_word_timestamps(
                segments=current_segments,
                model=model,
                tokenizer=tokenizer,
                mel=mel_segment,
                num_frames=segment_size,
                prepend_punctuations=prepend_punctuations,
                append_punctuations=append_punctuations,
                last_speech_timestamp=last_speech_timestamp,
            )

            if not single_timestamp_ending:
                last_word_end = get_end(current_segments)
                if last_word_end is not None and last_word_end > time_offset:
                    seek = round(last_word_end * FRAMES_PER_SECOND)

            # skip silence before possible hallucinations
            if hallucination_silence_threshold is not None:
                threshold = hallucination_silence_threshold
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        remaining_duration = window_end_time - last_word_end
                        if remaining_duration > threshold:
                            seek = round(last_word_end * FRAMES_PER_SECOND)
                        else:
                            seek = previous_seek + segment_size

                # if first segment might be a hallucination, skip leading silence
                first_segment = next_words_segment(current_segments)
                if first_segment is not None and is_segment_anomaly(first_segment):
                    gap = first_segment["start"] - time_offset
                    if gap > threshold:
                        seek = previous_seek + round(gap * FRAMES_PER_SECOND)
                        continue

                # skip silence before any possible hallucination that is surrounded
                # by silence or more hallucinations
                hal_last_end = last_speech_timestamp
                for si in range(len(current_segments)):
                    segment = current_segments[si]
                    if not segment["words"]:
                        continue
                    if is_segment_anomaly(segment):
                        next_segment = next_words_segment(
                            current_segments[si + 1 :]
                        )
                        if next_segment is not None:
                            hal_next_start = next_segment["words"][0]["start"]
                        else:
                            hal_next_start = time_offset + segment_duration
                        silence_before = (
                            segment["start"] - hal_last_end > threshold
                            or segment["start"] < threshold
                            or segment["start"] - time_offset < 2.0
                        )
                        silence_after = (
                            hal_next_start - segment["end"] > threshold
                            or is_segment_anomaly(next_segment)
                            or window_end_time - segment["end"] < 2.0
                        )
                        if silence_before and silence_after:
                            seek = round(
                                max(time_offset + 1, segment["start"])
                                * FRAMES_PER_SECOND
                            )
                            if content_duration - segment["end"] < threshold:
                                seek = content_frames
                            current_segments[si:] = []
                            break
                    hal_last_end = segment["end"]

            last_word_end = get_end(current_segments)
            if last_word_end is not None:
                last_speech_timestamp = last_word_end
        """


        """
        if verbose:
            for segment in current_segments:
                start, end, text = segment["start"], segment["end"], segment["text"]
                line = f"[{format_timestamp(start)} --> {format_timestamp(end)}] {text}"
                print(make_safe(line))
        """





        """
        Add the current clip's segment and token information to
        ALL_SEGMENTS
        ALL_TOKENS



        Clear segments that are instantaneous (start and end time the same)

        Clear segments that have no text
        """
        # if a segment is instantaneous or does not contain text, clear it
        """
        # CODE Change: current_segments_list now contains all the sub-segments
        # of the current segment for each hypothesis
        for i, segment in enumerate(current_segments):
            if segment["start"] == segment["end"] or segment["text"].strip() == "":
                segment["text"] = ""
                segment["tokens"] = []
                segment["words"] = []
        """
        for segments in current_segments_list[j]:
            for segment in segments:
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

        """
        # CODE Change: Code addition. Now that we have populated the current_segments_list for the
        # FIRST mel segment/segment, and cleaned it,
        # we can then collect the tokens for each hypothesis from the segment
        # and place it into current_tokens_list
        """
        # populate current_tokens_list for this hypothesis 
        try:
            current_tokens_list[j].extend([token for segment in current_segments for token in segment["tokens"]])
        except IndexError:
            current_tokens_list.append([token for segment in current_segments for token in segment["tokens"]])


    # This code is not used anymore in the new ver
    """
    all_segments is a list that contains all segments from all clips

    Add the processed segments from current_segments to all_segments
    Iterate over current_segments, starting off the enumeration from len(all_segments)
    to continue the ID from where it left of

    [{"id":1, {segment1} }, {"id":2, {segment2} },]

    Collect ALL segment information into a single list
    """
    """
    all_segments.extend(
        [
            {"id": i, **segment}
            for i, segment in enumerate(
                current_segments, start=len(all_segments)
            )
        ]
    )
    """
    """
    all_tokens accumulates all tokens from each iteration
    Add all tokens from current_segments to the overall list of tokens all_tokens
    """
    """
    all_tokens.extend(
        [token for segment in current_segments for token in segment["tokens"]]
    )
    """
    """
    Where to use the prompts at
    """
    """
    if not condition_on_previous_text or result.temperature > 0.5:
        # do not feed the prompt tokens if a high temperature was used
        prompt_reset_since = len(all_tokens)
    """
    # update progress bar
    # CODE CHANGE: No longer using progress bar
    # pbar.update(min(content_frames, seek) - previous_seek)





    """
    CODE CHANGE: CODE ADDITION

    This code basically replaces the code above. However, it is put separate because
    all our hypothesis may now start from different seek positions, depending on 
    how their individual segments were processed earlier due to different hypothesis 
    tokens

    Still the same logic
    """
    # loop through seek values corresponding to hypotheses
    # s_index will have the same range as number of hypotheses
    for s_index in range(len(seeks)):
        seek = seeks[s_index]
        while seek < content_frames:
            # GET THE NEW SEGMENT INFO
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[:, seek : seek + N_FRAMES]
            segment_size = min(N_FRAMES, content_frames - seek)
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES).to(model.device).to(dtype)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result: DecodingResult = decode_with_fallback(mel_segment) # Decode the new mel segment based on the curr seek position of the hypothesis
            hypothesis = result.tokens[s_index] # get corresponding hypothesis
            hypothesis = torch.tensor(hypothesis)

            current_segments = []

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if (
                    logprob_threshold is not None
                    and result.avg_logprob > logprob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            timestamp_tokens: torch.Tensor = hypothesis.ge(tokenizer.timestamp_begin)
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = torch.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0]
            consecutive.add_(1)
           
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(hypothesis))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = hypothesis[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = (
                        sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    )
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        hypothesis[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = hypothesis[timestamp_tokens.nonzero().flatten()]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = (
                        timestamps[-1].item() - tokenizer.timestamp_begin
                    )
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=hypothesis,
                        result=result,
                    )
                )
                seek += segment_size

            # if a segment is instantaneous or does not contain text, clear it
            for segments in current_segments:
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []
        
            current_segments_list[s_index].extend([current_segments])
            current_tokens_list[s_index].extend([token for segment in current_segments for token in segment["tokens"]])





    """
    Return the transcripted text, all the segment information etc.
    """
    """
    CODE CHANGE: We now return a different format
    return dict(
        text=tokenizer.decode(all_tokens[len(initial_prompt_tokens) :]),
        segments=all_segments,
        language=language,
    )
    """

    # loop through each hypothesis
    out_dicts = []
    for all_toks, segs in zip(current_tokens_list, current_segments_list):
        segs_list = [segment for sublist in segs for segment in sublist]
        out_dicts.append(dict(text=tokenizer.decode(all_toks[len(initial_prompt_tokens) :]), segments=segs_list, language=language))

    return out_dicts


def cli():
    from . import available_models

    def valid_model_name(name):
        if name in available_models() or os.path.exists(name):
            return name
        raise ValueError(
            f"model should be one of {available_models()} or path to a model checkpoint"
        )

    # fmt: off
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="audio file(s) to transcribe")
    parser.add_argument("--model", default="small", type=valid_model_name, help="name of the Whisper model to use")
    parser.add_argument("--model_dir", type=str, default=None, help="the path to save model files; uses ~/.cache/whisper by default")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for PyTorch inference")
    parser.add_argument("--output_dir", "-o", type=str, default=".", help="directory to save the outputs")
    parser.add_argument("--output_format", "-f", type=str, default="all", choices=["txt", "vtt", "srt", "tsv", "json", "all"], help="format of the output file; if not specified, all available formats will be produced")
    parser.add_argument("--verbose", type=str2bool, default=True, help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default=None, choices=sorted(LANGUAGES.keys()) + sorted([k.title() for k in TO_LANGUAGE_CODE.keys()]), help="language spoken in the audio, specify None to perform language detection")

    parser.add_argument("--temperature", type=float, default=0, help="temperature to use for sampling")
    parser.add_argument("--best_of", type=optional_int, default=5, help="number of candidates when sampling with non-zero temperature")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="number of beams in beam search, only applicable when temperature is zero")
    parser.add_argument("--patience", type=float, default=None, help="optional patience value to use in beam decoding, as in https://arxiv.org/abs/2204.05424, the default (1.0) is equivalent to conventional beam search")
    parser.add_argument("--length_penalty", type=float, default=None, help="optional token length penalty coefficient (alpha) as in https://arxiv.org/abs/1609.08144, uses simple length normalization by default")

    parser.add_argument("--suppress_tokens", type=str, default="-1", help="comma-separated list of token ids to suppress during sampling; '-1' will suppress most special characters except common punctuations")
    parser.add_argument("--initial_prompt", type=str, default=None, help="optional text to provide as a prompt for the first window.")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=True, help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
    parser.add_argument("--fp16", type=str2bool, default=True, help="whether to perform inference in fp16; True by default")

    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=0.2, help="temperature to increase when falling back when the decoding fails to meet either of the thresholds below")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="if the gzip compression ratio is higher than this value, treat the decoding as failed")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="if the average log probability is lower than this value, treat the decoding as failed")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="if the probability of the <|nospeech|> token is higher than this value AND the decoding has failed due to `logprob_threshold`, consider the segment as silence")
    parser.add_argument("--word_timestamps", type=str2bool, default=False, help="(experimental) extract word-level timestamps and refine the results based on them")
    parser.add_argument("--prepend_punctuations", type=str, default="\"\'“¿([{-", help="if word_timestamps is True, merge these punctuation symbols with the next word")
    parser.add_argument("--append_punctuations", type=str, default="\"\'.。,，!！?？:：”)]}、", help="if word_timestamps is True, merge these punctuation symbols with the previous word")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="(requires --word_timestamps True) underline each word as it is spoken in srt and vtt")
    parser.add_argument("--max_line_width", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of characters in a line before breaking the line")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="(requires --word_timestamps True) the maximum number of lines in a segment")
    parser.add_argument("--max_words_per_line", type=optional_int, default=None, help="(requires --word_timestamps True, no effect with --max_line_width) the maximum number of words in a segment")
    parser.add_argument("--threads", type=optional_int, default=0, help="number of threads used by torch for CPU inference; supercedes MKL_NUM_THREADS/OMP_NUM_THREADS")
    parser.add_argument("--clip_timestamps", type=str, default="0", help="comma-separated list start,end,start,end,... timestamps (in seconds) of clips to process, where the last end timestamp defaults to the end of the file")
    parser.add_argument("--hallucination_silence_threshold", type=optional_float, help="(requires --word_timestamps True) skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected")
    # fmt: on

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    model_dir: str = args.pop("model_dir")
    output_dir: str = args.pop("output_dir")
    output_format: str = args.pop("output_format")
    device: str = args.pop("device")
    os.makedirs(output_dir, exist_ok=True)

    if model_name.endswith(".en") and args["language"] not in {"en", "English"}:
        if args["language"] is not None:
            warnings.warn(
                f"{model_name} is an English-only model but receipted '{args['language']}'; using English instead."
            )
        args["language"] = "en"

    temperature = args.pop("temperature")
    if (increment := args.pop("temperature_increment_on_fallback")) is not None:
        temperature = tuple(np.arange(temperature, 1.0 + 1e-6, increment))
    else:
        temperature = [temperature]

    if (threads := args.pop("threads")) > 0:
        torch.set_num_threads(threads)

    from . import load_model

    model = load_model(model_name, device=device, download_root=model_dir)

    writer = get_writer(output_format, output_dir)
    word_options = [
        "highlight_words",
        "max_line_count",
        "max_line_width",
        "max_words_per_line",
    ]
    if not args["word_timestamps"]:
        for option in word_options:
            if args[option]:
                parser.error(f"--{option} requires --word_timestamps True")
    if args["max_line_count"] and not args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")
    if args["max_words_per_line"] and args["max_line_width"]:
        warnings.warn("--max_words_per_line has no effect with --max_line_width")
    writer_args = {arg: args.pop(arg) for arg in word_options}
    for audio_path in args.pop("audio"):
        try:
            result = transcribe(model, audio_path, temperature=temperature, **args)
            writer(result, audio_path, **writer_args)
        except Exception as e:
            traceback.print_exc()
            print(f"Skipping {audio_path} due to {type(e).__name__}: {str(e)}")


if __name__ == "__main__":
    cli()