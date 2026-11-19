"""Match trained voice unit boundaries and decode actions without sending payloads to TTS."""

import json
import sys


def configure_speech_completion(duplex, final_max_tokens):
    """Give a completed text unit room to finish speech before native reset."""
    native_generate = duplex.model.tts.generate_chunk

    def _generate_chunk(*args, **kwargs):
        if duplex.current_turn_ended:
            kwargs["max_new_token"] = max(kwargs["max_new_token"], final_max_tokens)
        return native_generate(*args, **kwargs)

    duplex.model.tts.generate_chunk = _generate_chunk


def observe_tts_termination(tts, generate, *args, **kwargs):
    """Observe native loop locals without changing its sampling or return values."""
    termination = {"termination_reason": "not_exposed_by_native_api"}
    previous = sys.getprofile()

    def _observe(frame, event, result):
        if previous is not None:
            previous(frame, event, result)
        if event != "return" or frame.f_code.co_name != "generate_chunk":
            return
        else:
            values = frame.f_locals
        if values.get("self") is tts and "finish" in values and "t" in values:
            ended = bool(values["finish"].all().item())
            step = int(values["t"])
            termination.update(
                termination_reason="speech_eos" if ended else "token_budget_exhausted",
                sampled_steps=step + 1,
                speech_eos_reached=ended,
                effective_max_new_tokens=int(values["max_new_token"]),
                final_sampled_token_ids=values["new_tokens"][:, step]
                .detach()
                .cpu()
                .reshape(-1)
                .tolist(),
            )

    sys.setprofile(_observe)
    try:
        tokens, cache = generate(*args, **kwargs)
    finally:
        sys.setprofile(previous)
    return tokens, cache, termination


def reset_speech_turn(duplex, *, close_language_turn=False):
    """Discard unfinished speech without touching input audio or LLM history."""
    if close_language_turn and not duplex.current_turn_ended:
        # The interrupted reply never emitted turn_eos. Merely clearing the TTS
        # cache leaves its continuation in the language model's context. Append
        # an empty, terminated speech unit at this safe unit boundary, without
        # rewinding KV or advancing the input audio clock.
        ids = [
            duplex.unit_token_id,
            duplex.speak_token_id,
            duplex.turn_eos_token_id,
            duplex.chunk_eos_token_id,
            int(duplex.tokenizer.convert_tokens_to_ids("</unit>")),
        ]
        duplex.decoder.feed(duplex.decoder.embed_tokens(ids))
        duplex.total_ids.extend(ids)
        duplex.total_hidden.append([])
        duplex.decoder.register_unit_end(
            input_type="audio",
            generated_tokens=ids[1:-1],
            is_listen=False,
            generated_text="",
        )
    duplex.current_turn_ended = True
    duplex.tts_text_start_pos = 0
    duplex.tts_past_key_values = None
    duplex.tts_current_turn_start_time = None
    duplex._reset_token2wav_for_new_turn()


def generate_control_action(
    duplex,
    native_generate,
    generation_kwargs,
    max_query_tokens=192,
    *,
    allow_interruption=False,
    native_action_policy=False,
):
    tokenizer = duplex.tokenizer
    action_ids = {
        name: int(tokenizer.convert_tokens_to_ids(f"<|{name}|>"))
        for name in ("listen", "speak", "delegate")
    }
    if duplex.pending_logits is None:
        return {"action": "listen", "is_listen": True, "text": "", "end_of_turn": False}
    elif native_action_policy and (
        duplex.is_session_stop_set()
        or duplex.is_break_set()
        or duplex._streaming_generate_count < duplex.force_listen_count
    ):
        return native_generate(**generation_kwargs)
    elif not bool(duplex.current_turn_ended) and not allow_interruption:
        return native_generate(**generation_kwargs)
    else:
        vector = duplex.pending_logits.reshape(-1, duplex.pending_logits.shape[-1])[-1]
    scores = {name: float(vector[token].item()) for name, token in action_ids.items()}
    if native_action_policy:
        # Select with the checkpoint decoder, including its repetition history.
        # Raw-logit argmax can otherwise lock a finished user turn into Listen.
        native_decode = duplex.decoder.decode
        first_token = native_decode(
            logits=duplex.pending_logits,
            mode=generation_kwargs.get("decode_mode", "sampling"),
            temperature=generation_kwargs.get("temperature", 0.7),
            top_k=generation_kwargs.get("top_k", 100),
            top_p=generation_kwargs.get("top_p", 0.8),
            listen_top_k=generation_kwargs.get("listen_top_k"),
            listen_prob_scale=generation_kwargs.get("listen_prob_scale", 1.0),
            text_repetition_penalty=generation_kwargs.get(
                "text_repetition_penalty", 1.05
            ),
            text_repetition_window_size=generation_kwargs.get(
                "text_repetition_window_size", 512
            ),
        )
        selected = {token: name for name, token in action_ids.items()}.get(
            int(first_token.item()), "speak"
        )
    else:
        selected = max(scores, key=scores.__getitem__)
    interrupted = not bool(duplex.current_turn_ended) and selected != "speak"
    if interrupted:
        # The native decoder replaces listen with tts_bos while a turn is open.
        # Every training unit has its own action label, including listen after
        # an unfinished speak. Release only the speech state before decoding
        # that selected action; retain the accumulated input and LLM KV cache.
        reset_speech_turn(duplex)
    if selected != "delegate":
        if native_action_policy:
            # Native generation consumes the already selected first token once;
            # every subsequent token still goes through the unchanged decoder.
            first_pending = True

            def _decode(*args, **kwargs):
                nonlocal first_pending
                if first_pending:
                    first_pending = False
                    return first_token
                else:
                    return native_decode(*args, **kwargs)

            duplex.decoder.decode = _decode
            try:
                result = native_generate(**generation_kwargs)
            finally:
                duplex.decoder.decode = native_decode
        else:
            forced = duplex.pending_logits.clone()
            forced[..., action_ids[selected]] = forced.max() + 100.0
            duplex.pending_logits = forced
            result = native_generate(**generation_kwargs)
        result["action"] = selected
        result["speech_interrupted"] = interrupted
        result["action_scores"] = scores
        return result
    else:
        duplex.pending_logits = None
    delegate_id = action_ids["delegate"]
    duplex.total_ids.append(delegate_id)
    logits, _ = duplex.decoder.feed(
        duplex.decoder.embed_token(delegate_id), return_logits=True
    )
    payload_ids = []
    for token_id in tokenizer.encode('{"query":"', add_special_tokens=False):
        payload_ids.append(int(token_id))
        logits, _ = duplex.decoder.feed(
            duplex.decoder.embed_token(int(token_id)), return_logits=True
        )
    query_ids = []
    forbidden = {
        *action_ids.values(),
        int(duplex.chunk_eos_token_id),
        int(duplex.turn_eos_token_id),
        int(tokenizer.convert_tokens_to_ids("</unit>")),
        int(duplex.tts_bos_token_id),
    }
    for _ in range(max_query_tokens):
        next_id = int(
            duplex.decoder.decode(
                logits=logits,
                mode="greedy",
                temperature=0.7,
                top_k=100,
                top_p=0.8,
                listen_prob_scale=1.0,
                text_repetition_penalty=1.05,
                text_repetition_window_size=512,
            ).item()
        )
        piece = tokenizer.decode([next_id], skip_special_tokens=False)
        if next_id in forbidden or any(
            char in piece for char in ('"', "\\", "\n", "\r", "{", "}")
        ):
            break
        else:
            query_ids.append(next_id)
            payload_ids.append(next_id)
        logits, _ = duplex.decoder.feed(
            duplex.decoder.embed_token(next_id), return_logits=True
        )
    query_text = tokenizer.decode(query_ids, skip_special_tokens=True).strip()
    for token_id in tokenizer.encode('"}', add_special_tokens=False):
        payload_ids.append(int(token_id))
        logits, _ = duplex.decoder.feed(
            duplex.decoder.embed_token(int(token_id)), return_logits=True
        )
    chunk_eos, unit_end = (
        int(duplex.chunk_eos_token_id),
        int(tokenizer.convert_tokens_to_ids("</unit>")),
    )
    duplex.decoder.feed(duplex.decoder.embed_token(chunk_eos))
    duplex.decoder.feed(duplex.decoder.embed_token(unit_end))
    duplex.total_ids.extend(payload_ids + [chunk_eos, unit_end])
    duplex.current_turn_ended = True
    duplex.total_hidden.append([])
    text = json.dumps({"query": query_text}, ensure_ascii=False, separators=(",", ":"))
    duplex.decoder.register_unit_end(
        input_type=duplex.current_mode.lower() if duplex.current_mode else "audio",
        generated_tokens=payload_ids,
        is_listen=False,
        generated_text=text,
    )
    return {
        "action": "delegate",
        "is_listen": False,
        "text": text,
        "audio_waveform": None,
        "end_of_turn": True,
        "n_tokens": len(payload_ids),
        "speech_interrupted": interrupted,
        "action_scores": scores,
    }


def inject_tool_result(duplex, event, *, batch=False):
    ids = duplex.tokenizer.encode(event, add_special_tokens=False)
    if batch:
        duplex.decoder.feed(duplex.decoder.embed_tokens(ids))
        duplex.total_ids.extend(int(token_id) for token_id in ids)
    else:
        for token_id in ids:
            duplex.decoder.feed(duplex.decoder.embed_token(int(token_id)))
            duplex.total_ids.append(int(token_id))
    return ids


def align_voice_prefill(duplex, tokenizer, add_timestamps):
    """Match the voice collator's <unit> timestamp audio_start/audio_end layout.

    The native AUDIO prefill feeds <unit> and then one audio embedding tensor.
    Action logits must be taken after audio_end, as they are during training.
    Vision prefill and the original action/JSON decoder are unchanged.
    """
    original_prefill = duplex.streaming_prefill
    original_feed = duplex.decoder.feed
    audio_start = int(tokenizer.convert_tokens_to_ids("<|audio_start|>"))
    audio_end = int(tokenizer.convert_tokens_to_ids("<|audio_end|>"))
    state = {"inside": False, "feeds": 0, "unit": 0, "timestamp_ids": []}

    def _feed(embeddings, *args, **kwargs):
        if not state["inside"]:
            return original_feed(embeddings, *args, **kwargs)
        state["feeds"] += 1
        if state["feeds"] == 1:
            result = original_feed(embeddings, *args, **kwargs)
            for token_id in state["timestamp_ids"]:
                original_feed(duplex.decoder.embed_token(token_id))
                duplex.total_ids.append(token_id)
            return result
        original_feed(duplex.decoder.embed_token(audio_start))
        original_feed(embeddings, *args, **kwargs)
        return original_feed(
            duplex.decoder.embed_token(audio_end),
            return_logits=kwargs.get("return_logits", False),
        )

    def _prefill(audio_waveform, **kwargs):
        state["unit"] += 1
        state["inside"], state["feeds"] = True, 0
        state["timestamp_ids"] = (
            tokenizer.encode(
                f"<{state['unit']:.1f} seconds>\n", add_special_tokens=False
            )
            if add_timestamps
            else []
        )
        try:
            result = original_prefill(audio_waveform=audio_waveform, **kwargs)
        finally:
            state["inside"] = False
        if result["success"]:
            audio_marker = duplex._current_unit_prefill_tokens[-1]
            duplex._current_unit_prefill_tokens = [
                duplex.unit_token_id,
                *state["timestamp_ids"],
                audio_start,
                audio_marker,
                audio_end,
            ]
        return result

    duplex.decoder.feed = _feed
    duplex.streaming_prefill = _prefill
