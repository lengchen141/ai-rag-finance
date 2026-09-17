import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://ws-ndaimedbdbwxttvo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": "你是一个专业的金融分析师，擅长用通俗的语言分析股票。"},
        {"role": "user", "content": "帮我分析一下贵州茅台(600519)最近的投资价值，从基本面和估值两个角度简要分析，300字以内。"}
    ]
)



print("=== AI 金融分析师 ===\n")
print(response.choices[0].message.content)