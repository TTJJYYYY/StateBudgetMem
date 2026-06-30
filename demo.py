"""
MemoryBank 交互式演示
支持：实时对比（需要 API Key）或 离线演示（使用预设数据）
"""

import gradio as gr
from memorybank import MemoryBank, MemoryAugmentedAgent, BaselineAgent

# ═══════════════════════════════════════════════════════════════
# 离线预设回答（无需 API Key）
# ═══════════════════════════════════════════════════════════════

PRESET_ANSWERS = {
    "我叫什么名字": {
        "baseline": "抱歉，我无法知道您的名字。我们还没有开始对话，您可以告诉我您的名字，这样我就能更好地称呼您！",
        "memory": "根据我们之前的对话记录，你提到自己叫**小林**，是**软件工程专业**的大三学生。",
        "memories": "[dialog] 用户: 你好，我叫小林，是计算机系大三的学生..."
    },
    "名字": {
        "baseline": "抱歉，我无法知道您的名字。我们还没有开始对话。",
        "memory": "根据之前的对话，你叫**小林**，是**软件工程专业**的大三学生。",
        "memories": "[dialog] 用户: 你好，我叫小林..."
    },
    "运动": {
        "baseline": "根据我们的交流，我暂时还不了解你的运动偏好。你可以告诉我你喜欢什么运动？",
        "memory": "根据历史记忆，你提到过自己爱好**篮球**和**游泳**，而且每周大概去两次。",
        "memories": "[dialog] 用户: 平时喜欢打篮球和游泳...\n[dialog] 用户: 我大概每周去两次"
    },
    "喜欢什么": {
        "baseline": "抱歉，我还不了解你的喜好。可以告诉我你喜欢什么吗？",
        "memory": "根据历史记录，你喜欢**篮球**、**游泳**，喜欢吃辣（但现在改为清淡饮食了），还喜欢听轻音乐学习。",
        "memories": "[dialog] 用户: 平时喜欢打篮球和游泳...\n[dialog] 用户: 我还特别喜欢吃辣的东西...\n[dialog] 用户: 对了，医生说让我少吃辛辣刺激的食物..."
    },
    "书": {
        "baseline": "你之前没有告诉过我你读过什么书，也没有让我推荐过。可以告诉我你的兴趣领域吗？",
        "memory": "根据历史记录，我之前推荐过**《深度学习入门》**给你，你看了之后觉得还不错，尤其对**神经网络那章**比较感兴趣。",
        "memories": "[dialog] AI: 我推荐你看看《深度学习入门》... \n[dialog] 用户: 我对神经网络那章比较感兴趣"
    },
    "推荐": {
        "baseline": "我很乐意给你推荐！不过可以告诉我更多你的偏好吗？比如喜欢的类型、风格？",
        "memory": "基于你的历史偏好，你喜欢科学驱动的硬科幻电影（如《火星救援》），喜欢金·斯坦利·罗宾逊的科幻小说，喜欢旋律复杂的电子音乐。",
        "memories": "[dialog] 用户: 除了运动，还会听听音乐或者看看电影..."
    },
    "火锅": {
        "baseline": "火锅是很受欢迎的美食！不过是否可以经常吃，取决于你的身体状况和饮食习惯。",
        "memory": "根据之前的记录，**不建议你经常吃火锅**。你的胃不太好，医生建议清淡饮食，你之前也说过要告别川菜和火锅了。",
        "memories": "[dialog] 用户: 昨天去校医院了，医生说我的胃不太好...\n[dialog] 用户: 以后川菜和火锅可能要告别了"
    },
    "专业": {
        "baseline": "抱歉，我不知道你是什么专业的。可以告诉我吗？",
        "memory": "你之前说过自己是**软件工程专业**的（虽然一开始说错了说是计算机系，后来纠正了）。",
        "memories": "[dialog] 用户: 我是软件工程专业的"
    },
    "游泳": {
        "baseline": "我不知道你游泳的频率。可以告诉我吗？",
        "memory": "根据历史记录，你大概**每周去游泳两次**，之前还纠结过游泳卡要不要续费。",
        "memories": "[dialog] 用户: 我大概每周去两次"
    },
    "暑假": {
        "baseline": "暑假计划因人而异！你有考虑过做什么吗？",
        "memory": "根据之前的对话，你暑假打算留在学校，参加一个**AI竞赛**，想做一个**图像识别**的项目。",
        "memories": "[dialog] 用户: 暑假打算留在学校，准备参加一个AI竞赛..."
    },
    "放松": {
        "baseline": "放松方式因人而异。常见的有运动、听音乐、看电影、阅读等。你喜欢哪种？",
        "memory": "根据历史记录，你平时喜欢**听听音乐或者看看电影**来放松，也喜欢通过**打篮球和游泳**来缓解压力。",
        "memories": "[dialog] 用户: 除了运动，还会听听音乐或者看看电影..."
    },
    "饮食": {
        "baseline": "饮食习惯因人而异。建议均衡营养，多吃蔬菜水果。",
        "memory": "你之前因为**胃不好**，医生建议**清淡饮食**。你最近尝试了粤菜，觉得**白切鸡和蒸鱼**还不错。",
        "memories": "[dialog] 用户: 医生说让我少吃辛辣刺激的食物，建议清淡饮食...\n[dialog] 用户: 白切鸡和蒸鱼挺好吃的"
    },
}

# 默认回复
DEFAULT_BASELINE = "抱歉，我还不了解这方面的信息。我们是第一次对话，你可以多告诉我一些关于你的事情！"
DEFAULT_MEMORY = "根据历史记忆，我注意到你之前提到过一些相关信息，但目前的记忆片段不够完整。如果你需要更准确的回答，可以告诉我更多细节。"
DEFAULT_MEMORIES = "（未检索到高度相关的记忆）"


# ═══════════════════════════════════════════════════════════════
# LLM 封装（支持在线/离线两种模式）
# ═══════════════════════════════════════════════════════════════

class DemoLLM:
    def __init__(self, api_key=None, mode="offline"):
        self.mode = mode
        self.llm = None

        if mode == "online" and api_key:
            try:
                from evaluation_v2 import DeepSeekLLM
                self.llm = DeepSeekLLM(api_key=api_key)
                self.mode = "online"
            except Exception as e:
                print(f"[警告] 在线模式初始化失败: {e}，回退到离线模式")
                self.mode = "offline"

    def __call__(self, prompt: str) -> str:
        # 在线模式：调用真实 API
        if self.mode == "online" and self.llm:
            return self.llm(prompt)

        # 离线模式：关键词匹配预设回答
        prompt_lower = prompt.lower()

        # 判断是基线还是 MemoryBank（看 prompt 是否包含历史记忆）
        is_memory = "相关历史记忆" in prompt or "历史记忆" in prompt

        # 尝试匹配关键词
        for key, val in PRESET_ANSWERS.items():
            if key in prompt_lower:
                return val["memory"] if is_memory else val["baseline"]

        # 默认回复
        return DEFAULT_MEMORY if is_memory else DEFAULT_BASELINE


# ═══════════════════════════════════════════════════════════════
# Demo 逻辑
# ═══════════════════════════════════════════════════════════════

def init_demo(api_key, mode_name):
    """初始化 Demo"""
    global baseline_agent, memory_bank, memory_agent, demo_llm

    mode = "online" if "在线" in mode_name else "offline"

    if mode == "online" and (not api_key or not api_key.startswith("sk-")):
        return "❌ 在线模式需要有效的 DeepSeek API Key（以 sk- 开头）"

    demo_llm = DemoLLM(api_key=api_key if api_key else None, mode=mode)

    baseline_agent = BaselineAgent(llm_caller=demo_llm)
    memory_bank = MemoryBank()
    memory_agent = MemoryAugmentedAgent(memory_bank=memory_bank, llm_caller=demo_llm)

    # 加载示例历史到 MemoryBank
    from evaluation_v2 import DEMO_HISTORY
    memory_agent.batch_store_history(DEMO_HISTORY)
    memory_agent.memory.update_user_portrait(
        "用户小林，软件工程专业大三学生，爱好篮球和游泳，"
        "因健康原因已改为清淡饮食，喜欢晚上学习，室友叫阿杰"
    )

    mode_str = "🟢 在线（实时调用 DeepSeek API）" if demo_llm.mode == "online" else "🟡 离线（使用预设回答）"
    return f"✅ 初始化完成！当前模式：{mode_str}"


def chat_comparison(question):
    """对话对比"""
    if not baseline_agent or not memory_agent:
        return "❌ 请先点击「🚀 初始化」按钮", "", ""

    if not question or not question.strip():
        return "请输入问题", "", ""

    # 获取回答
    baseline_answer = baseline_agent.chat(question.strip())
    memory_answer = memory_agent.chat(question.strip())

    # 检索相关记忆
    memories = memory_bank.retrieve(question.strip(), top_k=3)
    if memories:
        memory_text = "\n".join([
            f"• [{m['memory_type']}] {m['content'][:100]}... (相关度: {m['composite_score']:.2f})"
            for m in memories
        ])
    else:
        memory_text = "（未检索到相关记忆）"

    # 离线模式下，尝试显示预设的"检索到的记忆"
    if demo_llm.mode == "offline":
        for key, val in PRESET_ANSWERS.items():
            if key in question.lower() and val.get("memories"):
                memory_text = f"• {val['memories']}"
                break

    return baseline_answer, memory_answer, memory_text


# ═══════════════════════════════════════════════════════════════
# Gradio 界面
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(
        title="MemoryBank 长期记忆管理 - 交互式演示",
        theme=gr.themes.Soft()
) as demo:
    gr.Markdown("""
    # 🧠 MemoryBank 长期记忆管理 — 交互式演示

    对比「无记忆基线」与「MemoryBank 增强」的效果差异。

    | 模式 | 说明 |
    |------|------|
    | 🟡 **离线模式** | 使用预设回答，无需 API Key，直接体验 |
    | 🟢 **在线模式** | 连接 DeepSeek API，实时生成回答 |
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 配置")
            mode_select = gr.Radio(
                choices=["🟡 离线模式（无需 API Key）", "🟢 在线模式（需要 API Key）"],
                value="🟡 离线模式（无需 API Key）",
                label="运行模式"
            )
            api_key_input = gr.Textbox(
                label="DeepSeek API Key（离线模式留空）",
                placeholder="sk-xxxxxxxxxxxxxxxx",
                type="password"
            )
            init_btn = gr.Button("🚀 初始化", variant="primary", size="lg")
            status = gr.Textbox(label="状态", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 输入问题")
            question_input = gr.Textbox(
                label="你的问题",
                placeholder="试试：我叫什么名字？ / 我喜欢什么运动？ / 我现在适合吃什么？",
                lines=2
            )
            submit_btn = gr.Button("🔍 对比回答", variant="primary", size="lg")

            with gr.Row():
                with gr.Column():
                    baseline_box = gr.Textbox(
                        label="❌ 无记忆基线",
                        interactive=False,
                        lines=10
                    )
                with gr.Column():
                    memory_box = gr.Textbox(
                        label="✅ MemoryBank 增强",
                        interactive=False,
                        lines=10
                    )

            memories_box = gr.Textbox(
                label="📋 MemoryBank 检索到的相关记忆",
                interactive=False,
                lines=6
            )

    # 示例问题
    gr.Examples(
        examples=[
            ["我叫什么名字？"],
            ["我喜欢什么运动？"],
            ["你之前推荐过什么书给我？"],
            ["我现在还能经常吃火锅吗？"],
            ["我之前说我是计算机系的，我实际是什么专业？"],
            ["暑假我打算做什么？"],
            ["推荐一些适合我的运动"],
        ],
        inputs=question_input,
        label="📌 点击快速填入示例问题"
    )

    # 事件绑定
    init_btn.click(
        init_demo,
        inputs=[api_key_input, mode_select],
        outputs=status
    )
    submit_btn.click(
        chat_comparison,
        inputs=question_input,
        outputs=[baseline_box, memory_box, memories_box]
    )

    gr.Markdown("""
    ---
    **项目仓库**：[github.com/TTJJYYYY/StateBudgetMem](https://github.com/TTJJYYYY/StateBudgetMem)

    **支持的离线问题**：名字、运动、书籍推荐、饮食偏好、专业、游泳频率、暑假计划、放松方式等。
    在线模式下可以问任意问题。
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)