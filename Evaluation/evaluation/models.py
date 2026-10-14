from openai import OpenAI, AsyncOpenAI
import base64
import mimetypes

def encode_video_to_base64(video_path):
    mime_type, _ = mimetypes.guess_type(video_path)
    if not mime_type:
        mime_type = "video/mp4" 
    
    with open(video_path, "rb") as video_file:
        binary_data = video_file.read()
        base64_string = base64.b64encode(binary_data).decode('utf-8')
    
    return f"data:{mime_type};base64,{base64_string}"

MAX_TOKENS=8192

class Qwen3_API:
    def __init__(self, node,port, model):
        base_url = f"http://{node}:{port}/v1"
        self.client = AsyncOpenAI(
            api_key="EMPTY", 
            base_url=base_url, 
            timeout=3600
        )
        self.model_name = model

    async def __call__(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )
        result = response.choices[0].message.content
        return result

class Qwen3VL_API:
    def __init__(self, node,port, model):
        base_url = f"http://{node}:{port}/v1"
        self.client = AsyncOpenAI(
            api_key="EMPTY", 
            base_url=base_url, 
            timeout=3600
        )
        self.model_name = model

    async def __call__(self, prompt, video_path=None):
        if video_path:
            video_data_url = encode_video_to_base64(video_path)
            messages = [{
                "role": "user", 
                "content": [
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                    {"type": "text", "text": prompt}
                ]
            }]
        else:
            messages = [{"role": "user", "content": prompt}]
            
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.8,
            extra_body={
                "top_k": 20,
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            },
        )
        result = response.choices[0].message.content
        return result

from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
import torch

class Qwen3VL:
    def __init__(self, model_name="Qwen/Qwen3-VL-235B-A22B-Instruct", device_map="auto"):
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = self.model.device
    
    def __call__(self, prompt, image=None, max_new_tokens=MAX_TOKENS):
        content = []
        if image:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": prompt})
        
        messages = [{"role": "user", "content": content}]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]

class Qwen3:
    def __init__(self, model_name="Qwen/Qwen3-30B-A3B-Instruct-2507", device_map="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=device_map
        )
        self.device = self.model.device
    
    def __call__(self, prompt, max_new_tokens=MAX_TOKENS):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return content