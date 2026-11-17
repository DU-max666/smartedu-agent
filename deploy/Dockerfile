# 慧学平台 API 镜像：离线 mock 模式可直接运行；接入真实模型/Milvus 时配置环境变量
FROM python:3.10-slim

WORKDIR /srv/smartedu-agent

# 依赖层与代码层分离，便于缓存
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts
COPY eval ./eval

# 非 root 运行；数据卷挂载点（sqlite 演示库/日志）
RUN useradd --create-home huixue && chown -R huixue /srv/smartedu-agent
USER huixue

EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
