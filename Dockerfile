FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535
WORKDIR /work
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONDONTWRITEBYTECODE=1
# 默认运行完整业务演示；资料体检仍可用
# docker run --rm recovery-data python -m recovery check
CMD ["python", "-m", "recovery", "demo"]
