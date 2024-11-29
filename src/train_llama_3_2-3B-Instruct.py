# Import necessary libraries
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from datasets import load_dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
#import torch

# Set parameters
max_seq_length = 2048  # Adjust as needed
dtype = None  # None for auto-detection
load_in_4bit = True  # Use 4-bit quantization to reduce memory usage

# Choose the appropriate model
model_name = "unsloth/Llama-3.2-3B-Instruct"  # 3B parameter model

# Load the model and tokenizer with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Set up LoRA fine-tuning with memory optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Low-rank adaptation parameter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,  # 0 is optimized
    bias="none",     # "none" is optimized
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Apply the chat template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# Define a function to format prompts
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

# Load and preprocess your dataset
dataset = load_dataset("GeiserX/jw-instruct", split="train")

# If your dataset has 'messages' instead of 'conversations', rename the key
def rename_key(example):
    if 'messages' in example:
        example['conversations'] = example.pop('messages')
    return example

dataset = dataset.map(rename_key)

# Standardize and format the dataset
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)

# Configure the trainer with VRAM constraints in mind
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,  # Can speed up training for short sequences
    args=TrainingArguments(
        per_device_train_batch_size=1,  # Lower batch size to reduce VRAM usage
        gradient_accumulation_steps=16,  # Increase to maintain effective batch size
        warmup_steps=5,
        max_steps=1000,  # Set this for a full training run
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",  # Use this if you don't want to report to W&B or similar
        gradient_checkpointing=True,  # Enable gradient checkpointing
    ),
)

# Train on responses only to optimize training
trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# Start training
trainer_stats = trainer.train()

# After training, enable model for inference
FastLanguageModel.for_inference(model)  # Enable optimized inference

# Save the trained model locally
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# Optionally, you can push the model to Hugging Face Hub if needed
# model.push_to_hub("your_username/lora_model", token="your_hf_token")
# tokenizer.push_to_hub("your_username/lora_model", token="your_hf_token")

# Prepare the model for inference on the target machine
# Quantize and save the model in GGUF format with 4-bit quantization
model.save_pretrained_gguf(
    "model_gguf",   # Directory to save the model
    tokenizer,
    quantization_method="q4_0",  # 4-bit quantization suitable for CPU inference
)

# The model is now saved in 'model_gguf' directory and ready to be used with LocalAI

# Sample code to generate text (can be used for testing)
# Prepare an input prompt
messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")  # Use 'cpu' if not using a GPU

# Generate output
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs.input_ids,
    streamer=text_streamer,
    max_new_tokens=128,
    use_cache=True,
    temperature=1.5,
    top_p=0.9,
)

# Note: Adjust 'max_new_tokens', 'temperature', and 'top_p' as needed