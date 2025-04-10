import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SequentialChain
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("请在环境变量中设置 DEEPSEEK_API_KEY")

# 初始化 ChatDeepSeek 模型
llm = ChatDeepSeek(
    model="deepseek-chat",  # 或者使用其他可用的模型名称
    temperature=0.7,
    max_tokens=512,
    timeout=30,
    max_retries=2,
    api_key=api_key,  # 如果未设置环境变量，可以在此直接提供 API 密钥
)

# # 构建消息列表
# messages = [
#     SystemMessage(content="Translate the following from English into Italian"),
#     HumanMessage(content="hi are you ok?")
# ]

# Step 1: 翻译英文文章为中文
translate_prompt = ChatPromptTemplate.from_template(
    "请将以下英文内容翻译成中文：\n\n{article}"
)
translate_chain = LLMChain(llm=llm, prompt=translate_prompt, output_key="translated")
# Step 2: 为中文文章生成标题（15 字以内）
title_prompt = ChatPromptTemplate.from_template(
    "以下是中文文章内容：{translated}\n请为这篇文章起一个吸引人的中文标题（不超过15个字）："
)
title_chain = LLMChain(llm=llm, prompt=title_prompt, output_key="title")



article_chain = SequentialChain(
    chains=[translate_chain, title_chain],
    input_variables=["article"],
    output_variables=["translated", "title"],
    verbose=True
)
# 示例英文输入
article_text = """
LangChain is an open-source framework that simplifies building applications with large language models. 
It supports features like memory, chaining, agents, and retrieval, allowing developers to compose complex workflows easily.
"""

# 执行链式处理
result = article_chain.invoke({"article": article_text})

# 输出结果
print("\n📘 中文翻译：\n", result["translated"])
print("\n🏷️ 生成标题：", result["title"])
