<!---
Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<h1 align="center"> <p LoRTA 🤗 PEFT</p></h1>
<h3 align="center">
    <p> Tensor Low rank adapters (LoRTA) implementation using HF's PEFT</p>
</h3>

PEFT is integrated with Transformers for easy model training and inference, Diffusers for conveniently managing different adapters, and Accelerate for distributed training and inference for really big models.

> [!TIP]
> Visit the [PEFT](https://huggingface.co/PEFT) organization to read about the PEFT methods implemented in the library and to see notebooks demonstrating how to apply these methods to a variety of downstream tasks. Click the "Watch repos" button on the organization page to be notified of newly implemented methods and notebooks!

I also included experiment scripts from the [VeRA paper](https://arxiv.org/abs/2310.11454) openreview submission.

## Quickstart

Install PEFT from pip:

```bash
pip install -r requirements.txt
pip install .
```

Try an example:
    
```bash
python instruct/lora_llama2_7b.sh
```