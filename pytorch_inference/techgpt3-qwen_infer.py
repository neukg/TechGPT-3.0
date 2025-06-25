import json
import torch
import uvicorn
import threading
import time
from fastapi import FastAPI, Request
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests

app = FastAPI()

# ✅ 模型路径（请替换为你自己的模型路径）
model_name = "your model path"
device = "cuda"

print("🔄 加载模型中...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ 模型加载完成")

@app.post("/question_answer")
async def create_item(request: Request):
    json_post_raw = await request.json()
    json_post_list = json.loads(json.dumps(json_post_raw))

    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history') or []
    if prompt is None:
        return {"response": "Prompt不能为空", "history": []}

    system_prompt = [{"role": "system", "content": "You are a helpful assistant."}]
    current_prompt = [{"role": "user", "content": prompt}]
    messages = system_prompt + history + current_prompt
    messages = [m for m in messages if m.get("content") is not None]

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception as e:
        return {"response": f"Chat模板渲染错误: {str(e)}", "history": history}

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # ✅ 使用思考模式推荐参数进行生成
    generated_ids = model.generate(
        **model_inputs,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0,
        max_new_tokens=4096,
        do_sample=True  # 禁止贪婪解码
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    history = history + current_prompt
    history.append({"role": "assistant", "content": response_text})
    print("📤 Chat response:", response_text)
    return {"response": response_text, "history": history}

def call_api():
    time.sleep(5)  # 等待服务器启动
    print("🚀 开始测试调用 API...")

    # ✅ 测试用 API 地址（请替换为实际端口）
    url = "http://localhost:your_port/question_answer"
    payload = {
        "prompt": "市政府决定从2025年7月起，全面推行垃圾分类制度。请将上述通告扩展为一则正式完整的新闻通稿，内容包括背景、措施与意义。",
        "history": []
    }

    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print("✅ 模型回答:", result["response"])
    else:
        print("❌ 调用失败:", response.text)

if __name__ == '__main__':
    # ✅ 启动服务器线程（请替换为你的端口号）
    server_thread = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=your_port, log_level="error"),
        daemon=True
    )
    server_thread.start()

    # 启动客户端调用
    call_api()
