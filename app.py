"""An image generation agent implemented by assistant with qwq"""

import os

from agents.search_agent import SearchAgent
from llm.oai import TextChatAtOAI
from gui.web_ui import WebUI
import datetime

DR_MODEL_NAME = os.getenv("DR_MODEL_NAME", "")
DR_MODEL_SERVER = os.getenv("DR_MODEL_SERVER", "")
DR_MODEL_API_KEY = os.getenv("DR_MODEL_API_KEY", "")

ROOT_RESOURCE = os.path.join(os.path.dirname(__file__), 'resource')

def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")

def init_dev_search_agent_service(name: str = 'SEARCH', port: int = 8002, desc: str = '初版', reasoning: bool = True, max_llm_calls: int = 20, tools = ['search', 'visit'], addtional_agent = None):
    llm_cfg = TextChatAtOAI({
        'model': DR_MODEL_NAME,
        'model_type': 'oai',
        'model_server': DR_MODEL_SERVER,
        'api_key': DR_MODEL_API_KEY,
        'generate_cfg': {
            'fncall_prompt_type': 'nous',
            'temperature': 0.85,
            'top_p': 0.95,
            'top_k': -1,
            'presence_penalty': 1.1,
            'max_tokens': 32768,
            'stream_options': {
                'include_usage': True,
            },
            'timeout': 3000
        },
    })
    def make_system_prompt():
        system_message = "You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields. For every request, synthesize information from credible, diverse sources to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.\n\n"
        return system_message
    
    bot = SearchAgent(
        llm=llm_cfg,
        function_list=tools,
        system_message="",
        name=f'Tongyi DeepResearch',
        description=f"I am Tongyi DeepResearch, a leading open-source Deep Research Agent, welcome to try!",
        extra={
            'reasoning': reasoning,
            'max_llm_calls': max_llm_calls,
        },
        addtional_agent = addtional_agent,
        make_system_prompt = make_system_prompt,
        custom_user_prompt=''''''
    )

    return bot



def app_gui():
    agents = []
    for name, port, desc, reasoning, max_llm_calls, tools in [
        ('Tongyi DeepResearch', 8004, '...', True, 50, ['search', 'visit', 'google_scholar', 'PythonInterpreter']),
    ]:
        search_bot_dev = init_dev_search_agent_service(
            name=name,
            port=port,
            desc=desc,
            reasoning=reasoning,
            max_llm_calls=max_llm_calls,
            tools=tools,
        )
        agents.append(search_bot_dev)


    chatbot_config = {
        'prompt.suggestions': [
            '中国国足的一场比赛，国足首先失球，由一名宿姓球员扳平了。后来还发生了点球。比分最终是平均。有可能是哪几场比赛',
            'When is the paper submission deadline for the ACL 2025 Industry Track, and what is the venue address for the conference?',
            'On June 6, 2023, an article by Carolyn Collins Petersen was published in Universe Today. This article mentions a team that produced a paper about their observations, linked at the bottom of the article. Find this paper. Under what NASA award number was the work performed by R. G. Arendt supported by?',
            '有一位华语娱乐圈的重要人物，与其兄弟共同创作并主演了一部在中国南方沿海城市上映的喜剧电影，这部电影成为该类型的开山之作。与此同时，这位人物还凭借两首极具影响力的本地方言歌曲在音乐领域取得突破，极大推动了本地方言流行音乐的发展。请问，这一切发生在20世纪70年代的哪一年？',
            '有一首欧洲国家的国歌自20世纪50年代初被正式采用，并只选用了其中的一部分歌词。同一年，一位中国文艺界的重要人物创作了一部以民间传说为基础的戏曲作品，并在当年担任了多个文化领域的重要职务。请问这位中国文艺界人物是谁？',
            '有一部英国文坛上极具影响力的长篇诗歌，由一位16世纪末的著名诗人创作，这位诗人在16世纪90年代末于伦敦去世后，被安葬在一个象征英国文学传统的著名场所，与多位文学巨匠为邻。请问，这位诗人安息之地是哪里？',
            '出一份三天两夜的端午北京旅游攻略',
            '对比下最新小米汽车和保时捷性能参数，然后根据最终的结果分析下性价比最高的车型，并给出杭州的供应商',
            '量子计算突破对现有加密体系的威胁',
            '人工智能伦理框架的全球差异',
            '老龄化社会对全球养老金体系的长期冲击',
            '全球碳中和目标下的能源转型路径差异',
            '塑料污染在海洋食物链中的累积效应',
            'AI生成内容（如AI绘画）对传统艺术价值的重构'
        ],
        'user.name': 'User',
        'verbose': True
    }
    messages = {'role': 'user', 'content': '介绍下你自己'}
    WebUI(
        agent=agents,
        chatbot_config=chatbot_config,
    ).run(
        message=messages,
        share=True,
        server_name='127.0.0.1',
        server_port=7860,
        concurrency_limit=20,
        enable_mention=False,
    )


if __name__ == '__main__':
    app_gui()
