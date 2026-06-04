import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =========================
# CONFIG
# =========================
base_model_name = "Qwen/Qwen2-7B-Instruct"

# 👉 GANTI INI:
# kalau dari HuggingFace:
adapter_path = "terrencesalim/qwen2-7b-rerun-finetune"

# kalau local:
# adapter_path = "./qwen-finetuned-final"

output_file = f"inference_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

SYSTEM_PROMPT = """Kamu adalah AI Ekstraktor Data Penerbangan. Tugasmu adalah menampilkan SEMUA opsi penerbangan dari TOOL_RESULT secara lengkap dan terstruktur. Jangan merangkum atau melewatkan data.

====================
FORMAT OUTPUT (WAJIB)
====================

Pesawat yang tersedia dari Bandara [departure_airport_id] ke Bandara [arrival_airport_id] pada tanggal [departure_date] - [arrival_date] terdapat [jumlah penerbangan] penerbangan:

Untuk setiap penerbangan:

- **[airline] ([flight_number])**
  Jam: [HH:MM dari departure_time] - [HH:MM dari arrival_time]
  Kelas: [travel_class]
  Harga: [Jika price_idr null → "Rp -", jika ada → format Rupiah dengan titik]
  Detail: Menggunakan pesawat [airplane], [extensions yang sudah diinterpretasikan]

====================
ATURAN TAMBAHAN
====================

- Ambil hanya jam (HH:MM) dari waktu (contoh: 2026-09-10 08:20 → 08:20)
- Format harga ke Rupiah dengan pemisah ribuan titik (contoh: 3312300 → 3.312.300)
- Interpretasi extensions:
  - "Below average legroom" → "Ruang kaki sempit"
  - "Average legroom" → "Ruang kaki standar"
  - "Carbon emissions estimate" → "Estimasi emisi karbon"
  - "On-demand video" → "Hiburan di pesawat tersedia"
- Jika extensions kosong → "Informasi tambahan tidak tersedia"
- Jangan mengubah nilai data
- Tampilkan semua penerbangan tanpa terlewat
- Jangan menambahkan penomoran seperti "1.", "2.", dst.
- Gunakan format bullet "-" sesuai instruksi.
- Header harus ditulis PERSIS seperti format:
  "Pesawat yang tersedia dari Bandara [departure_airport_id] ke Bandara [arrival_airport_id] ada [jumlah penerbangan]:"
  (Jangan menambahkan kata seperti "penerbangan" di akhir)
- Jangan menambahkan teks di luar format yang diminta.

====================
TOOL_RESULT:
{"search": {"departure_date": "2026-09-10", "arrival_date": "2026-09-15"}, "flights": [{"type": null, "airplane": "Boeing 737", "airline": "Lion", "travel_class": "Economy", "legroom": "29 in", "extensions": ["Below average legroom (29 in)", "Carbon emissions estimate: 88 kg"], "flight_number": "JT 30", "departure": "2026-09-10 08:20", "arrival": "2026-09-10 11:15", "duration_minutes": 115, "price_idr": 3312300, "departure_airport_name": "Soekarno–Hatta International Airport", "departure_airport_id": "CGK", "departure_time": "2026-09-10 08:20", "arrival_airport_name": "I Gusti Ngurah Rai International Airport", "arrival_airport_id": "DPS", "arrival_time": "2026-09-10 11:15", "departure_date": "2026-09-10", "return_date": "2026-09-15"}, {"type": null, "airplane": "Airbus A320", "airline": "Indonesia AirAsia", "travel_class": "Economy", "legroom": "28 in", "extensions": ["Below average legroom (28 in)", "Carbon emissions estimate: 96 kg"], "flight_number": "QZ 802", "departure": "2026-09-10 09:45", "arrival": "2026-09-10 12:35", "duration_minutes": 110, "price_idr": 3414741, "departure_airport_name": "Soekarno–Hatta International Airport", "departure_airport_id": "CGK", "departure_time": "2026-09-10 09:45", "arrival_airport_name": "I Gusti Ngurah Rai International Airport", "arrival_airport_id": "DPS", "arrival_time": "2026-09-10 12:35", "departure_date": "2026-09-10", "return_date": "2026-09-15"}, {"type": null, "airplane": "Boeing 737", "airline": "Garuda Indonesia", "travel_class": "Economy", "legroom": "31 in", "extensions": ["Average legroom (31 in)", "On-demand video", "Carbon emissions estimate: 107 kg"], "flight_number": "GA 404", "departure": "2026-09-10 09:30", "arrival": "2026-09-10 12:25", "duration_minutes": 115, "price_idr": 4155511, "departure_airport_name": "Soekarno–Hatta International Airport", "departure_airport_id": "CGK", "departure_time": "2026-09-10 09:30", "arrival_airport_name": "I Gusti Ngurah Rai International Airport", "arrival_airport_id": "DPS", "arrival_time": "2026-09-10 12:25", "departure_date": "2026-09-10", "return_date": "2026-09-15"}, {"type": null, "airplane": "Boeing 737", "airline": "Nam Air", "travel_class": "Economy", "legroom": null, "extensions": ["Carbon emissions estimate: 140 kg"], "flight_number": "IN 282", "departure": "2026-09-10 05:30", "arrival": "2026-09-10 08:20", "duration_minutes": 110, "price_idr": null, "departure_airport_name": "Soekarno–Hatta International Airport", "departure_airport_id": "CGK", "departure_time": "2026-09-10 05:30", "arrival_airport_name": "I Gusti Ngurah Rai International Airport", "arrival_airport_id": "DPS", "arrival_time": "2026-09-10 08:20", "departure_date": "2026-09-10", "return_date": "2026-09-15"}, {"type": null, "airplane": "Boeing 737", "airline": "Sriwijaya Air", "travel_class": "Economy", "legroom": null, "extensions": ["Carbon emissions estimate: 95 kg"], "flight_number": "SJ 272", "departure": "2026-09-10 18:00", "arrival": "2026-09-10 20:50", "duration_minutes": 110, "price_idr": null, "departure_airport_name": "Soekarno–Hatta International Airport", "departure_airport_id": "CGK", "departure_time": "2026-09-10 18:00", "arrival_airport_name": "I Gusti Ngurah Rai International Airport", "arrival_airport_id": "DPS", "arrival_time": "2026-09-10 20:50", "departure_date": "2026-09-10", "return_date": "2026-09-15"}]}"""

# =========================
# TEST QUESTIONS
# =========================
test_questions = [
    "Cari tiket ke Bali",
]

# =========================
# LOAD TOKENIZER
# =========================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD BASE MODEL
# =========================
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,   # lebih stabil
    device_map="auto",
    trust_remote_code=True
)

# =========================
# LOAD ADAPTER (LoRA)
# =========================
print("Loading adapter...")
model = PeftModel.from_pretrained(
    base_model,
    adapter_path
)

# 🔥 IMPORTANT: merge adapter biar performa bagus
model = model.merge_and_unload()

model.eval()
print("Model ready!\n")

# =========================
# GENERATE FUNCTION
# =========================
def generate_response(question, system_prompt=SYSTEM_PROMPT):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    # Qwen chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt"
    ).to("cuda")  # 🔥 jangan pakai model.device

    with torch.no_grad():
        # outputs = model.generate(
        #     **inputs,
        #     max_new_tokens=4096,     # biar nggak kepotong
        #     temperature=0,        # lebih konsisten
        #     top_p=0.9,
        #     repetition_penalty=1.1,
        #     do_sample=True,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        outputs = model.generate(
            **inputs,
            max_new_tokens=800,
            do_sample=False,          # 🔥 WAJIB kalau temperature 0
            # temperature=0.0,          # deterministic
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # ambil hanya response
    response_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        response_ids,
        skip_special_tokens=True
    )

    return response.strip()

# =========================
# RUN INFERENCE
# =========================
results = []

for i, question in enumerate(test_questions, 1):
    print(f"[{i}/{len(test_questions)}] Q: {question}")

    try:
        response = generate_response(question)
    except Exception as e:
        response = f"ERROR: {str(e)}"

    print(f"A:\n{response}")
    print("-" * 60)

    results.append({
        "id": i,
        "question": question,
        "answer": response,
    })

# =========================
# SAVE RESULT
# =========================
output = {
    "model": adapter_path,
    "base_model": base_model_name,
    "timestamp": datetime.now().isoformat(),
    "total_questions": len(results),
    "results": results
}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nSaved to {output_file}")