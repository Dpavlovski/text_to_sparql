import time

import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel

models_to_benchmark = [
    "intfloat/e5-small-v2",  # 77 rank, 33M
    "sentence-transformers/all-MiniLM-L12-v2",  # 100 rank , 33M
    "intfloat/multilingual-e5-small",  # 34 rank , 118M
    "intfloat/multilingual-e5-base",  # 30 rank, 278M
    "intfloat/multilingual-e5-large-instruct",  # 3 rank, 568M
]

loaded_models = {}

print("🔹 Loading models...")
for model_name in models_to_benchmark:
    print(f"📥 Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cpu")
    model.eval()

    loaded_models[model_name] = (tokenizer, model)

print("\n✅ All models loaded successfully!")

sample_text = "This is a test sentence for benchmarking embeddings."

results = {}

print("\n⏳ Running inference benchmarks...")
for model_name, (tokenizer, model) in loaded_models.items():
    inputs = tokenizer(sample_text, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        _ = model(**inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()

    inference_time = end_time - start_time
    results[model_name] = inference_time
    print(f"⏱ {model_name}: {inference_time:.4f} seconds")

print("\n✅ Benchmarking completed!")

model_params = {
    "intfloat/e5-small-v2": "33M",
    "sentence-transformers/all-MiniLM-L12-v2": "33M",
    "intfloat/multilingual-e5-small": "118M",
    "intfloat/multilingual-e5-base": "278M",
    "intfloat/multilingual-e5-large-instruct": "568M"
}

print("\n📊 Model Parameters:")
print(f"{'Model Name':<40} {'Parameters':<10}")
print("-" * 50)
for model, params in model_params.items():
    short_name = model.split("/")[-1]
    print(f"{short_name:<40} {params:<10}")

plt.figure(figsize=(9, 5))
models = list(results.keys())
times = list(results.values())

labels = [f"{model.split('/')[-1]}\n({model_params[model]})" for model in models]

sorted_data = sorted(zip(times, models, labels), key=lambda x: x[0])
sorted_times = [x[0] for x in sorted_data]
sorted_labels = [x[2] for x in sorted_data]

colors = plt.cm.viridis_r([int(p[:-1]) / 1000 for p in model_params.values()])
bars = plt.barh(sorted_labels, sorted_times, height=0.6, color=colors)

plt.xlabel('Inference Time (seconds)', fontsize=10)
plt.title('Model Benchmark: Speed vs Parameter Size', fontsize=12, pad=15)
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.gca().invert_yaxis()

for bar, time, label in zip(bars, sorted_times, sorted_labels):
    params = label.split('\n')[-1].strip('()')
    plt.text(bar.get_width() + 0.08,
             bar.get_y() + bar.get_height() / 2,
             f'{time:.1f}s | {params}',
             va='center',
             ha='left',
             fontsize=9)

import matplotlib.patches as mpatches

handles = [
    mpatches.Patch(color=colors[0], label='Small (33M)'),
    mpatches.Patch(color=colors[2], label='Medium (118M)'),
    mpatches.Patch(color=colors[3], label='Large (278M)'),
    mpatches.Patch(color=colors[4], label='XL (568M)')
]
plt.legend(handles=handles, title='Model Size', loc='lower right')

plt.grid(axis='x', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()
