from threading import Thread
from typing import Iterator
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import time

MAX_NEW_TOKENS = 8000
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9

if torch.cuda.is_available():
    model_id = "/media/hdd2/models/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.use_default_system_prompt = False
    print("Model loaded - ready to generate text")


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = 50,
    repetition_penalty: float = 1.2,
) -> Iterator[str]:
    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})
    for user, assistant in chat_history:
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id,
    )
    print(max_new_tokens)
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    total_new_tokens = 0
    t1 = time.time()
    for text in streamer:
        new_tokens = len(tokenizer.encode(text))
        if total_new_tokens + new_tokens > max_new_tokens:
            break
        total_new_tokens += new_tokens
        yield text
    t2 = time.time()
    print("Inference took - ", t2 - t1, "seconds")
    print(f"Total new tokens created: {total_new_tokens}")


class ChatHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/v1/chat/completions":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(post_data)
            user_prompt = data["messages"][-1]["content"]
            chat_history = [(msg["content"], "") for msg in data["messages"][:-1]]
            system_prompt = next(
                (msg["content"] for msg in data["messages"] if msg["role"] == "system"),
                "You are a helpful assistant.",
            )
            max_new_tokens = int(data.get("max_tokens", MAX_NEW_TOKENS))
            temperature = float(data.get("temperature", DEFAULT_TEMPERATURE))
            top_p = float(data.get("top_p", DEFAULT_TOP_P))
            log_incoming_requests(user_prompt, chat_history)

            if "stream" in data and data["stream"]:
                self.send_response(200)
                self.send_header("Content-type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                # Emit the initial chunk with the assistant's role
                initial_chunk = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1694268190,
                    "model": "Llama-3-8B-instruct",
                    "system_fingerprint": "fp_44709d6fcb",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                self.wfile.write(
                    f"data: {json.dumps(initial_chunk)}\n\n".encode("utf-8")
                )
                self.wfile.flush()

                # Emit chunks with generated text
                for new_text in generate(
                    user_prompt,
                    chat_history,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                ):
                    text_chunk = {
                        "id": "chatcmpl-123",
                        "object": "chat.completion.chunk",
                        "created": 1694268190,
                        "model": "Llama-3-8B-instruct",
                        "system_fingerprint": "fp_44709d6fcb",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": new_text},
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                    self.wfile.write(
                        f"data: {json.dumps(text_chunk)}\n\n".encode("utf-8")
                    )
                    self.wfile.flush()

                # Emit the final chunk indicating completion
                final_chunk = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion.chunk",
                    "created": 1694268190,
                    "model": "Llama-3-8B-instruct",
                    "system_fingerprint": "fp_44709d6fcb",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
                self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()
                self.close_connection = True

            else:
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()

                outputs = []
                for new_text in generate(
                    user_prompt,
                    chat_history,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_p,
                ):
                    outputs.append(new_text)
                response = {
                    "id": "chatcmpl-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": data.get("model", "gpt-3.5-turbo-0125"),
                    "system_fingerprint": "fp_44709d6fcb",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "".join(outputs),
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(tokenizer.encode(user_prompt)),
                        "completion_tokens": len(tokenizer.encode("".join(outputs))),
                        "total_tokens": len(tokenizer.encode(user_prompt))
                        + len(tokenizer.encode("".join(outputs))),
                    },
                }
                self.wfile.write(json.dumps(response).encode("utf-8"))
                self.close_connection = True


def run(server_class=HTTPServer, handler_class=ChatHandler, port=5000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


def log_incoming_requests(user_prompt, chat_history):
    print("user_prompt", user_prompt)
    print("chat_history", chat_history)


if __name__ == "__main__":
    run()
