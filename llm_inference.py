import json
import picollm
import constants
import numpy as np
import input_speech
import output_speech

pllm = picollm.create(
    access_key=constants.ACCESS_KEY,
    model_path=constants.MODEL_PATH)

pllm_dialog = pllm.get_dialog(mode="chat",system="You are a voice assistant. Also be concise and do not ramble.")

prompt = ""
token_accumulation = []
generated_tokens = []
tts_buffer = ""
is_beginning = True

def tokenizer(text):
    global is_beginning
    print(f"Tokenizing text: {text}")
    if is_beginning:
        is_beginning = False
        tokens = pllm.tokenize(text, bos=True, eos=False)
    else:
        tokens = pllm.tokenize(text, bos=False, eos=False)
    token_accumulation.extend(tokens)
    return tokens

def forward_pass(tokens):
    for token in tokens:
        logits = pllm.forward(token)
        generated_tokens.append(int(np.argmax(logits)))

def generate_step(text):
    tokens = tokenizer(text)
    forward_pass(tokens)

def decode_tokens(tokens):
    return "".join(input_speech.id_to_token[token] for token in tokens)

def generate_final_step(text):
    tokens = pllm.tokenize(text, eos=True, bos=False)
    forward_pass(tokens)
    print(token_accumulation)
    print("Generated tokens:")
    print(decode_tokens(generated_tokens))
    global is_beginning
    is_beginning = True

def prompt_accumulation(text):
    global prompt
    prompt += text


def generate_res(text):
    global prompt
    prompt += text
    pllm_dialog.add_human_request(prompt)
    output_speech.open_audio_stream()
    # Define streaming callback
    def on_partial_token(text_chunk: str):
        output_speech.add_to_audio_queue(text_chunk=text_chunk)
    # Generate response from LLM
    res = pllm.generate(
        prompt=pllm_dialog.prompt(),
        completion_token_limit=30,
        temperature=0.6,        # natural but stable
        top_p=0.9,
        presence_penalty=0.3,
        frequency_penalty=0.2,
        stop_phrases={"\nUser:", "\nHuman:"},
        stream_callback=on_partial_token
    )
    output_speech.tts_flush()
    pllm_dialog.add_llm_response(res.completion)
    print("LLM Response:", res.completion)
    prompt = ""


def generate_res1(text):
    global prompt
    prompt += text
    pllm_dialog.add_human_request(prompt)
    # Define streaming callback
    def on_partial_token(text_chunk: str):
        global tts_buffer
        tts_buffer += text_chunk
        if any(p in tts_buffer for p in [".", "?", "!", ",", ";"]):
            speak_buffer()
    # Generate response from LLM
    res = pllm.generate(
        prompt=pllm_dialog.prompt(),
        completion_token_limit=30,
        temperature=0.6,        # natural but stable
        top_p=0.9,
        presence_penalty=0.3,
        frequency_penalty=0.2,
        stop_phrases={"\nUser:", "\nHuman:"},
        stream_callback=on_partial_token
    )
    if tts_buffer.strip():
        speak_buffer()
    output_speech.tts_flush()
    pllm_dialog.add_llm_response(res.completion)
    print("LLM Response:", res.completion)
    prompt = ""

def speak_buffer():
    global tts_buffer
    output_speech.tts_streaming_step(tts_buffer.removesuffix("\n<|endoftext|>"))
    tts_buffer = ""

