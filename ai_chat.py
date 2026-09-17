import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url="https://ws-ndaimedbdbwxttvo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

# 对话历史，这是多轮对话的关键
messages = [
    {"role": "system", "content": "你是一个专业的金融分析师，回答要简洁专业。"}
]

print("=== AI 金融分析师（多轮对话版）===")
print("输入 'quit' 退出\n")

while True:
    user_input = input("你: ")
    if user_input.lower() == 'quit':
        print("再见！")
        break

    # 把用户的问题加入对话历史
    messages.append({"role": "user", "content": user_input})

    # 流式调用
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
        stream=True   # 关键：开启流式输出
    )

    print("AI: ", end="", flush=True)
    reply = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            print(text, end="", flush=True)
            reply += text

    # 把 AI 的回复也加入历史，这样下一轮对话AI能记住上下文
    messages.append({"role": "assistant", "content": reply})
    print("\n")