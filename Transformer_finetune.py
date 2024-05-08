from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, GPT2LMHeadModel, TrainingArguments

model_id = "gpt2" # gpt2 gpt2-medium gpt2-large
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the padding token to the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_id)

# Re-initialize all the weights
model.init_weights()

dataset = load_dataset("Abirate/english_quotes", split="train")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3000,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
    learning_rate=2e-3
)

lora_config = LoraConfig(
    r=8,
    target_modules=["c_attn"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(model, lora_config)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="quote",
)

trainer.train()
