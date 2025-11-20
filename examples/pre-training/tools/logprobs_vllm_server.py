# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

import io
import argparse
import pickle

query_cache = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)


class PromptRequest(BaseModel):
    prompt_token_ids: list[list[int]]
    max_tokens: int = 1
    temperature: float = 0.7
    top_p: float = 0.95
    prompt_logprobs: int = -1
    logprobs: int = -1


def init_model(model_name, tensor_parallel_size: int = 1):
    global llm
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        trust_remote_code=True,
        max_logprobs=-1,
    )


@app.post("/generate")
async def generate_logit(request: PromptRequest):
    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        prompt_logprobs=request.prompt_logprobs,
        logprobs=request.logprobs,
        detokenize=False,
        n=1,
    )
    if tuple(request.prompt_token_ids[0]) in query_cache:
        buffer = io.BytesIO()
        pickle.dump(query_cache[tuple(request.prompt_token_ids[0])], buffer)
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="application/octet-stream")

    outputs = llm.generate(
        TokensPrompt(prompt_token_ids=request.prompt_token_ids[0]),
        sampling_params=sampling_params,
        use_tqdm=True,
    )
    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        all_token = []
        all_ids = []
        # prompt probs
        for d in range(len(prompt_logprobs[1:])):
            token_probs = []
            ids = sorted(prompt_logprobs[1:][d].keys())[: request.prompt_logprobs]
            for idx in ids:
                token_probs.append(prompt_logprobs[1:][d][idx].logprob)
            all_token.append(token_probs)
            all_ids.append(ids)
        # decode probs
        decode_probs = []
        decode_ids = sorted(output.outputs[0].logprobs[0])[: request.logprobs]
        for idx in decode_ids:
            decode_probs.append(output.outputs[0].logprobs[0][idx].logprob)
        all_token.append(decode_probs)
        all_ids.append(decode_ids)
    package = {
        "logits": all_token,
        "ids": all_ids,
    }
    query_cache[tuple(request.prompt_token_ids[0])] = package
    buffer = io.BytesIO()
    pickle.dump(package, buffer)
    buffer.seek(0)
    return Response(content=buffer.read(), media_type="application/octet-stream")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="model_path")
    parser.add_argument("--tp", type=int, default=1, help="Tensor_parallel")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host_ip")
    parser.add_argument("--port", type=int, default=8008, help="host_port")
    args = parser.parse_args()

    print(f"Loading Model: {args.model}")

    init_model(args.model, args.tp)

    print("Loaded Model !")

    uvicorn.run(app, host=args.host, port=args.port)
