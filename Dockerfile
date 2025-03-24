FROM python:3.10-slim
# 设置工作目录
WORKDIR /app
# 复制整个项目到容器中
COPY . .
# 安装系统依赖，减少不必要的软件包安装
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    freeglut3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p handtex/data/model && \
    curl -L -o handtex/data/model/handtex.safetensors https://github.com/VoxelCubes/Hand-TeX/releases/download/model/handtex.safetensors && \
    curl -L -o handtex/data/model/encodings.txt https://github.com/VoxelCubes/Hand-TeX/releases/download/model/encodings.txt

# 运行 Hand TeX 主程序
CMD ["sh", "-c", "PYTHONPATH=. python handtex/main.py"]