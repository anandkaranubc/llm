# LLaMA Model Fine-Tuning and Text Generation

This project demonstrates the process of fine-tuning a pre-trained LLaMA model using PEFT (Parameter-Efficient Fine-Tuning) techniques and generating text responses. The model is fine-tuned on a medical terms dataset and is capable of generating detailed responses to user prompts.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
   - [Setup](#setup)
   - [Fine-Tuning the Model](#fine-tuning-the-model)
   - [Generating Text](#generating-text)
3. [Configuration](#configuration)
4. [Dataset](#dataset)
5. [Model Details](#model-details)
6. [Troubleshooting](#troubleshooting)
7. [Acknowledgements](#acknowledgements)

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/llama-finetuning.git
cd llama-finetuning
pip install -q accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7
pip install huggingface_hub
```

## Usage

### Setup

1. **Import Necessary Modules:**

    ```python
    import torch
    from trl import SFTTrainer
    from peft import LoraConfig
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, pipeline)
    ```

2. **Load the Pre-trained Model:**

    ```python
    llama_model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2", 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, "float16"),
            bnb_4bit_quant_type="nf4"
        )
    )
    llama_model.config.use_cache = False
    llama_model.config.pretraining_tp = 1
    ```

3. **Load the Tokenizer:**

    ```python
    llama_tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path="aboonaji/llama2finetune-v2",
        trust_remote_code=True
    )
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"
    ```

### Fine-Tuning the Model

1. **Set Up Training Arguments:**

    ```python
    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        max_steps=100
    )
    ```

2. **Load the Dataset:**

    ```python
    train_dataset = load_dataset(
        path="aboonaji/wiki_medical_terms_llam2_format",
        split="train"
    )
    ```

3. **Define PEFT Configuration:**

    ```python
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1
    )
    ```

4. **Initialize and Train with SFTTrainer:**

    ```python
    llama_sft_trainer = SFTTrainer(
        model=llama_model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=llama_tokenizer,
        peft_config=peft_config,
        dataset_text_field="text"
    )
    llama_sft_trainer.train()
    ```

### Generating Text

1. **Define the User Prompt:**

    ```python
    user_prompt = "Tell me about scoliosis"
    ```

2. **Initialize Text Generation Pipeline:**

    ```python
    text_generation_pipeline = pipeline(
        task="text-generation", 
        model=llama_model, 
        tokenizer=llama_tokenizer, 
        max_length=300
    )
    ```

3. **Generate and Print the Model's Answer:**

    ```python
    model_answer = text_generation_pipeline(f"<s> [INST] {user_prompt} [/INST]")
    print(model_answer[0]['generated_text'])
    ```

## Configuration

- **Model:** `aboonaji/llama2finetune-v2`
- **Tokenizer:** `aboonaji/llama2finetune-v2`
- **Dataset:** `aboonaji/wiki_medical_terms_llam2_format`
- **Training Arguments:**
  - Output Directory: `./results`
  - Batch Size: 4
  - Max Steps: 100
- **PEFT Configuration:**
  - Task Type: `CAUSAL_LM`
  - r: 64
  - LoRA Alpha: 16
  - LoRA Dropout: 0.1

## Dataset

The dataset used for fine-tuning is a collection of medical terms in the LLaMA2 format. It is available at `aboonaji/wiki_medical_terms_llam2_format` and contains text data that helps the model learn and generate accurate medical information.

## Model Details

The LLaMA model used in this project is a causal language model fine-tuned with PEFT techniques. The model is configured to use 4-bit quantization, allowing for efficient training and inference on lower computational resources.

## Troubleshooting

- **Memory Issues:** Ensure your machine has sufficient memory to handle the dataset and model. Consider using cloud-based solutions if local resources are insufficient.
- **Dependency Conflicts:** Make sure all dependencies are installed with the specified versions to avoid conflicts.
- **Training Problems:** Double-check the dataset path and ensure the training arguments are set correctly.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing the transformer models and datasets.
- The [PEFT](https://github.com/microsoft/peft) library for parameter-efficient fine-tuning.
- The contributors and maintainers of the [LLaMA](https://github.com/facebookresearch/llama) project.
