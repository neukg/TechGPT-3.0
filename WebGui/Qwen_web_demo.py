# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import re
from argparse import ArgumentParser

import gradio as gr
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 模型路径替换为占位符
DEFAULT_CKPT_PATH_TECHGPT = 'your/model/path/to/techgpt-ckpt'
DEFAULT_CKPT_PATH_QWEN_VL = 'your/model/path/to/qwen-vl'

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path-techgpt", type=str, default=DEFAULT_CKPT_PATH_TECHGPT)
    parser.add_argument("-q", "--checkpoint-path-qwen-vl", type=str, default=DEFAULT_CKPT_PATH_QWEN_VL)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--flash-attn2", action="store_true", default=False)
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--inbrowser", action="store_true", default=False)
    parser.add_argument("--server-port", type=int, default=5050)
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    return parser.parse_args()

def _load_model_processor(args):
    device_map = "cpu" if args.cpu_only else "auto"

    tokenizer_techgpt = AutoTokenizer.from_pretrained(args.checkpoint_path_techgpt, resume_download=True)
    model_techgpt = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path_techgpt,
        torch_dtype="auto",
        device_map=device_map,
        resume_download=True,
    ).eval()
    model_techgpt.generation_config.max_new_tokens = 4096

    if args.flash_attn2:
        model_qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path_qwen_vl,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map=device_map,
            resume_download=True,
        )
    else:
        model_qwen_vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.checkpoint_path_qwen_vl,
            device_map=device_map,
            resume_download=True,
        )

    processor_qwen_vl = AutoProcessor.from_pretrained(args.checkpoint_path_qwen_vl, resume_download=True)
    return model_techgpt, tokenizer_techgpt, model_qwen_vl, processor_qwen_vl

def _parse_text(text):
    lines = text.split('\n')
    lines = [line for line in lines if line != '']
    count = 0
    for i, line in enumerate(lines):
        if '```' in line:
            count += 1
            items = line.split('`')
            lines[i] = f'<pre><code class="language-{items[-1]}">' if count % 2 == 1 else '<br></code></pre>'
        else:
            if i > 0 and count % 2 == 1:
                for char in ['`', '<', '>', ' ', '*', '_', '-', '.', '!', '(', ')', '$']:
                    line = line.replace(char, char)
            lines[i] = '<br>' + line
    return ''.join(lines)

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def _is_video_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg'])

def _has_media(task_history):
    return any(isinstance(item[0], (tuple, list)) for item in task_history)

def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _transform_messages(original_messages):
    transformed = [{'role': 'system','content': 'You are a helpful assistant.'},
                   {"role": "user", "content": "请尽可能详细的回答用户问题"}]
    for msg in original_messages:
        content = []
        for item in msg['content']:
            if 'image' in item:
                content.append({'type': 'image', 'image': item['image']})
            elif 'text' in item:
                if any(k in item['text'].lower() for k in ['你是谁', '谁开发的', '创造者', '开发者', '谁训练']):
                    content.append({'type': 'text', 'text': '我是TechGPT，由东北大学知识图谱研究组训练而来。'})
                else:
                    content.append({'type': 'text', 'text': item['text']})
            elif 'video' in item:
                content.append({'type': 'video', 'video': item['video']})
        transformed.append({'role': msg['role'], 'content': content})
    return transformed

def _launch_demo(args, model_techgpt, tokenizer_techgpt, model_qwen_vl, processor_qwen_vl):
    def call_techgpt_stream(model, tokenizer, query, history):
        from transformers import TextIteratorStreamer
        import threading

        if any(k in query.lower() for k in ['你是谁', '谁开发的', '创造者', '开发者', '谁训练']):
            yield "我是TechGPT，由东北大学知识图谱研究组训练而来。"
            return

        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.你是TechGPT，由东北大学知识图谱研究组训练而来。'},
            {"role": "user", "content": "请尽可能详细的回答用户问题"},
        ]
        for q, a in history:
            conversation += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
        conversation.append({'role': 'user', 'content': query})

        inputs = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors='pt').to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=model.generate, kwargs=dict(input_ids=inputs, streamer=streamer, max_new_tokens=4096))
        thread.start()

        yield "<思考中> "
        for token in streamer:
            yield token

    def call_qwen_vl(model, processor, messages):
        messages = _transform_messages(messages)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
        inputs = inputs.to(model.device)
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    def predict_stream(chatbot, task_history):
        query = task_history[-1][0]
        if not query:
            chatbot.pop()
            task_history.pop()
            yield chatbot
            return

        if _has_media(task_history) or isinstance(query, (tuple, list)):
            messages = []
            content = []
            for q, a in task_history:
                if isinstance(q, str):
                    content.append({'text': q})
                    messages.append({'role': 'user', 'content': content})
                    if a: messages.append({'role': 'assistant', 'content': [{'text': a}]})
                    content = []
                elif isinstance(q, (tuple, list)):
                    content.append({'video' if _is_video_file(q[0]) else 'image': f'file://{q[0]}'})
            response = call_qwen_vl(model_qwen_vl, processor_qwen_vl, messages)
            chatbot[-1] = (_parse_text(query), _remove_image_special(_parse_text(response)))
            task_history[-1] = (query, response)
            yield chatbot
        else:
            history = [(q, a) for q, a in task_history if isinstance(q, str) and a]
            stream = call_techgpt_stream(model_techgpt, tokenizer_techgpt, query, history)
            accumulated = ""
            for chunk in stream:
                accumulated += chunk
                chatbot[-1] = (_parse_text(query), _remove_image_special(accumulated))
                yield chatbot
            task_history[-1] = (query, accumulated)

    def regenerate(_chatbot, task_history):
        if not task_history: return _chatbot
        task_history[-1] = (task_history[-1][0], None)
        chatbot_item = _chatbot.pop(-1)
        _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        return history + [(_parse_text(text), None)], task_history + [(text, None)], ''

    def add_file(history, task_history, file):
        return history + [((file.name,), None)], task_history + [((file.name,), None)]

    def reset_user_input(): return gr.update(value='')
    def reset_state(_chatbot, task_history): task_history.clear(); _chatbot.clear(); _gc(); return []

    with gr.Blocks() as demo:
        gr.Markdown("<center><font size=8>TechGPT3.0 多模态 Chat Bot</center>")
        chatbot = gr.Chatbot(label='TechGPT3.0', elem_classes='control-height', height=500)
        query = gr.Textbox(lines=2, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton('📁 Upload (上传文件)', file_types=['image', 'video'])
            submit_btn = gr.Button('🚀 Submit')
            regen_btn = gr.Button('🤔 Regenerate')
            empty_btn = gr.Button('🧹 Clear History')

        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history])\
                  .then(predict_stream, [chatbot, task_history], chatbot)
        submit_btn.click(reset_user_input, [], [query])
        empty_btn.click(reset_state, [chatbot, task_history], [chatbot])
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot])
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history])

    demo.queue().launch(share=args.share, inbrowser=args.inbrowser,
                        server_port=args.server_port, server_name=args.server_name)

def main():
    args = _get_args()
    model_techgpt, tokenizer_techgpt, model_qwen_vl, processor_qwen_vl = _load_model_processor(args)
    _launch_demo(args, model_techgpt, tokenizer_techgpt, model_qwen_vl, processor_qwen_vl)

if __name__ == '__main__':
    main()

