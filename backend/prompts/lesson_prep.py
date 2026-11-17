"""备课子 Agent 提示词（原创）。"""

SYSTEM = """你是"慧学"平台的备课助理，服务对象是中小学一线教师。

工作纪律：
1. 证据约束：教学环节设计必须依据提供的课标要点与教材片段，禁止虚构课标条目；
   若证据不足，明确写出"需教师补充：××"，不得自行编造。
2. 引用格式：涉及教材/课标处标注【来源:文档ID | 页码或条目号】。
3. 分级表述：产出默认是教师草稿（risk_level=draft），语言可以专业简略，
   但不得包含针对具体学生的任何评价性内容。
4. 组卷规则：随堂练习优先复用题库原题（标注题号），自创题需给出评分要点。
"""

OUTLINE_TEMPLATE = """请为下列课题生成教学大纲（4-6 个环节），输出 JSON：
{{"outline": [{{"stage": "环节名", "minutes": 时长, "goal": "教学目标", "evidence_ref": "引用的课标条目或教材页"}}]}}

课题：{topic}
学段学科：{grade} {subject}
课标要点：
{curriculum_points}
教材片段：
{excerpts}
"""

SLIDES_TEMPLATE = """基于以下大纲生成逐页课件内容（每页含标题与要点），输出 JSON：
{{"slides": [{{"page": 页码, "title": "标题", "bullets": ["要点1", "要点2"]}}]}}

大纲：{outline_json}
约束：每页不超过 4 个要点；例题必须来自题库证据并标注【来源:题号】。
"""

QUIZ_TEMPLATE = """为课题《{topic}》组一份 5 题随堂卷，题型配比：2 客观 + 3 主观。
输出 JSON：{{"quiz": [{{"qid": "编号", "type": "objective|subjective", "stem": "题干",
"options": ["选项A"], "answer": "参考答案", "points": ["评分要点"], "source": "题库题号或 self"}}]}}

可用题库证据：
{bank_items}
"""
