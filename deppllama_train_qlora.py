import transformers
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import csv
import random
from typing import List
import json
import argparse

random.seed(23)


from deppllama_utils import *
 
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
 
#import fire
import torch
from datasets import load_dataset
import pandas as pd

#============================================
#               PARAMETERS
#============================================


parser = argparse.ArgumentParser(description='Train')

parser.add_argument('-t','--train_file_path', help='Input Train File Path',required=True)
parser.add_argument('-d','--dev_file_path', help='Input Dev File Path',required=True)
parser.add_argument('-m','--model_name', help='Model name', default="yahma/llama-7b-hf")
parser.add_argument('-o','--output_dir_path', help='Model name', default="model")
parser.add_argument('-e','--epochs', help='epochs', type=int, default=1)
parser.add_argument('-lr','--learning_rate', help='Learning Rate', type=float, default=3e-4)
parser.add_argument('-bs','--batch_size', help='batch size', type=int, default=32)
parser.add_argument('-mbs','--micro_batch_size', help='micro batch size', type=int, default=8)
parser.add_argument('--group_by_length', action="store_true", default=False)
parser.add_argument('-dq','--disable_qlora', action="store_true", default=False)

args = parser.parse_args()

print(args)

input_train_path=args.train_file_path
input_dev_path = args.dev_file_path
model_name = args.model_name
output_dir_path = args.output_dir_path
epochs = args.epochs
group_by_length = args.group_by_length

disable_qlora=args.disable_qlora

TOKENIZER_MODEL = model_name
BASE_MODEL = model_name
OUTPUT_DIR = output_dir_path


task = "*"

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = ['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head']


 
CUTOFF_LEN = 512
TRIM_LEN = 100000

EPOCHS = epochs
BATCH_SIZE = args.batch_size 
MICRO_BATCH_SIZE = args.micro_batch_size
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = args.learning_rate
WARMUP_RATIO=0.1

print("LEARNING_RATE:\t" + str(LEARNING_RATE))

tmp_train_file_name = "tmp_train.json"
tmp_dev_file_name = "tmp_dev.json"

#============================================
#               FUNCTIONS
#============================================

#LOAD INPUT TSV files 
def load(input_file_path):
    dataset_df = pd.read_csv(input_file_path, header=None, usecols=[0,1, 2, 3], names=['0', '1', '2', '3'], sep="\t", quoting=csv.QUOTE_NONE, encoding='utf-8').astype(str)
    dataset_df = dataset_df.rename(
        columns={"0": "id", "1": "prefix", "2": "input_text", "3": "target_text"}
    )
    #dataset_df["prefix"] = ""
    dataset_df = dataset_df[["id", "input_text", "target_text", "prefix"]]
    return dataset_df

 
# Notice: in the generate_and_tokenize_prompt function result["labels"] is rewritten
def tokenize(prompt, cutoff_len, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result

 
# Notice: result["labels"] is rewritten so that only the output is considered
def generate_and_tokenize_prompt(data_point, add_eos_token=True):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, CUTOFF_LEN)

    user_prompt = generate_prompt_str(
        data_point["input"]
    )
    tokenized_user_prompt = tokenize(
        user_prompt, CUTOFF_LEN, add_eos_token=add_eos_token
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    if add_eos_token:
        user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably
    return tokenized_full_prompt

    

def load_and_prepare_data(input_file_path: str, trim_len=15):

    df = load(input_file_path)

    dataset_data = [
        {
            "instruction": "Parse this sentence:",
            "input": row_dict["input_text"],
            "output": row_dict["target_text"]
        }
        for row_dict in df.to_dict(orient="records")
    ]

    for elem in dataset_data:
        osplit = elem["output"].split()
        l = len(osplit)
        if l > trim_len:
            isplit = elem["input"].split()[:trim_len]
            osplit = osplit[:trim_len]
            elem["input"] = " ".join(isplit)
            elem["output"] = " ".join(osplit)
        

    return dataset_data

def remove_example_by_length(lst, target_length):
    result = []
    for item in lst:
        if len(item["input_ids"])<target_length:
            result.append(item)
    return result

#============================================
#                   MAIN
#============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#-------------------
#    LOAD DATA 
#-------------------
train_data = load_and_prepare_data(input_train_path, TRIM_LEN)
dev_data = load_and_prepare_data(input_dev_path, TRIM_LEN)


with open(tmp_train_file_name, "w") as f:
   json.dump(train_data, f)
with open(tmp_dev_file_name, "w") as f:
   json.dump(dev_data, f)

json_train = load_dataset("json", data_files=tmp_train_file_name)
json_dev = load_dataset("json", data_files=tmp_dev_file_name)



#-------------------
#    LOAD MODEL
#-------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL, trust_remote_code=True)

# PREPARE DATA
train_data = ( json_train["train"].shuffle().map(generate_and_tokenize_prompt) )
val_data = ( json_dev["train"].shuffle().map(generate_and_tokenize_prompt) )


original_train_length = len(train_data)

train_data = remove_example_by_length(train_data, CUTOFF_LEN)

if(len(train_data)!=original_train_length):
    print("WARNING:")
    print("original_train_length: " + str(original_train_length))
    print("len(train_data): " + str(len(train_data)))

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

if not disable_qlora:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_4bit=True,
        quantization_config=quant_config,
        #torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0},
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.half()

if "falcon" in model_name:
    tokenizer.pad_token = tokenizer.eos_token
else:
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"

print("padding_side\t" + str(tokenizer.padding_side))


# PREPARE MODEL
model = prepare_model_for_kbit_training(model)

if "falcon" in model_name:
    config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )
else:
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_ratio=WARMUP_RATIO,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_strategy = "steps",
    logging_steps=1,
    optim="paged_adamw_32bit",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    output_dir=OUTPUT_DIR,
    save_total_limit=0,
    group_by_length=group_by_length,
    load_best_model_at_end=True,
    label_names=["labels"]
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False

if torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

if "falcon" in model_name:
    model.config.pad_token_id = model.config.eos_token_id
else:
    model.config.pad_token_id = 0
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

if torch.__version__ >= "2":
    print("YES! I AM 2.0 :-)")
    model = torch.compile(model)

#-------------------
#    LOAD MODEL
#-------------------

trainer.train()

model.save_pretrained(OUTPUT_DIR)