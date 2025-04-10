import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
# 加载环境变量
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
prompt = ChatPromptTemplate.from_template("""
你是一个资深的技术招聘专家，擅长分析英文简历。请根据以下简历内容，完成以下任务：

1. 提取应聘者姓名（如有）  
2. 提取工作年限  
3. 提取技术关键词（最多10个）  
4. 判断是否符合“高级后端工程师”职位要求，输出 Yes / No  
5. 给出详细评价理由（不少于3句）

请严格按照以下格式输出（保持英文）：
---
Name: <姓名>
Experience: <年限>
Skills: <技能关键词列表>
Match Senior Backend Role: <Yes/No>
Reason: <理由>
---
以下是简历内容：
{text}
""")

resume = """
Hi, my name is Michael Stone. I'm a software engineer with over 7 years of experience,
mostly working in backend systems. I’ve worked at companies like Uber and Dropbox,
focusing on distributed systems, microservices, and database performance.

My core skills include Java, Go, Redis, Kafka, Kubernetes, Docker, and PostgreSQL.
I recently led the architecture redesign of a real-time messaging platform at scale.
I’m now looking for more challenging backend roles in high-performance systems.
"""



chain = prompt | llm 

# 调用模型
response = chain.invoke({"text": resume})

# 输出模型的回复
print("🧠 DeepSeek 回复：", response.content)
