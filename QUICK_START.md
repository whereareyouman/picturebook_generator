# 🚀 快速开始指南 / Quick Start Guide

## 本地测试（推荐方式）/ Local Testing (Recommended)

### 方法一：使用 uv（推荐）/ Method 1: Using uv (Recommended)

#### 1. 进入项目目录 / Navigate to Project
```bash
cd /Users/laisingkuang/Downloads/gemini_picturebook_generator
```

#### 2. 安装依赖 / Install Dependencies
```bash
# 如果还没有安装 uv，先安装它
# If you don't have uv, install it first:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
# Install project dependencies
uv sync
```

#### 3. 配置 API 密钥 / Configure API Key
```bash
# 复制环境变量模板
# Copy environment template
cp .env.template .env

# 编辑 .env 文件，添加您的 Google API 密钥
# Edit .env file and add your Google API key from:
# https://aistudio.google.com/app/apikey
nano .env  # 或使用其他编辑器 / or use your preferred editor
```

在 `.env` 文件中设置：
```
GOOGLE_API_KEY=您的实际API密钥
```

#### 4. 测试 API 连接 / Test API Connection
```bash
# 测试 API 是否正常工作
# Test if API works correctly
uv run python test_api.py
```

如果看到 `🎉 All tests passed!`，说明配置成功！

#### 5. 启动 Web 界面 / Start Web Interface
```bash
uv run gemini-picturebook
# 然后在浏览器中打开 / Then open in browser:
# http://localhost:8080
```

### 方法二：使用 pip（传统方式）/ Method 2: Using pip (Traditional)

#### 1. 创建虚拟环境 / Create Virtual Environment
```bash
cd /Users/laisingkuang/Downloads/gemini_picturebook_generator
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
```

#### 2. 安装依赖 / Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. 配置 API 密钥 / Configure API Key
```bash
cp .env.template .env
# 编辑 .env 文件添加您的 API 密钥
# Edit .env file to add your API key
```

#### 4. 测试 API / Test API
```bash
python test_api.py
```

#### 5. 启动 Web 界面 / Start Web Interface
```bash
# 方式 1: 使用命令行工具
python -m gemini_picturebook_generator.run_ui

# 方式 2: 直接运行 Flask UI
python flask_ui.py

# 然后在浏览器中打开 / Then open in browser:
# http://localhost:8080
```

## 使用 Web 界面生成故事 / Generate Stories via Web Interface

1. **打开浏览器访问** / **Open browser to**: http://localhost:8080
2. **输入故事创意** / **Enter story idea**: 例如 "一只害羞的机器人学习画画"
3. **选择场景数量** / **Select number of scenes**: 建议首次测试选择 3-6 个场景
4. **选择艺术风格** / **Select art style**: 卡通、动漫、水彩等
5. **点击生成** / **Click Generate**: 等待生成完成（每个场景约 6 秒）

## Claude Desktop 集成（可选）/ Claude Desktop Integration (Optional)

### 添加到 Claude Desktop / Add to Claude Desktop

复制以下内容到您的 Claude Desktop 配置文件 (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "gemini-picturebook-generator": {
      "command": "uv",
      "args": [
        "--directory",
        "/Users/laisingkuang/Downloads/gemini_picturebook_generator",
        "run",
        "gemini-picturebook-mcp"
      ],
      "env": {
        "GOOGLE_API_KEY": "your_actual_api_key_here",
        "MCP_SERVER_MODE": "true"
      }
    }
  }
}
```

### 在 Claude 中使用 / Use with Claude

**生成第一个故事 / First Story:**
```
Generate a 3-scene cartoon story about a robot learning to bake cookies
```

**自定义故事 / Custom Story:**
```
Create a 12-scene watercolor story about Luna the dragon who's afraid of heights, set in a mountain kingdom
```

**浏览图库 / Browse Gallery:**
```
Show me my generated stories
```

**显示故事 / Display Story:**
```
Display story [story_id] as an artifact
```

## ✅ 成功指标 / Success Indicators

- ✅ 测试脚本通过所有检查 / Test script passes all checks
- ✅ MCP 服务器启动无错误 / MCP server starts without errors  
- ✅ Claude Desktop 显示服务器已连接 / Claude Desktop shows the server as connected
- ✅ 可以通过 Claude 生成故事 / You can generate stories through Claude
- ✅ Web 界面可以正常访问 / Web interface is accessible
- ✅ 可以成功生成故事 / Stories can be generated successfully

## 🛟 快速故障排查 / Quick Troubleshooting

**"API key not configured" / "API 密钥未配置"**: 
- 编辑 `.env` 文件，添加真实的 API 密钥
- Edit `.env` file with real API key

**"Connection failed" / "连接失败"**: 
- 验证 API 密钥：https://aistudio.google.com/app/apikey
- Verify API key at https://aistudio.google.com/app/apikey

**"Server not found" / "服务器未找到"**: 
- 检查 Claude Desktop 配置中的路径是否正确
- Check path in Claude Desktop config

**"Permission denied" / "权限被拒绝"**: 
- 运行 `chmod +x` 在项目目录上
- Run `chmod +x` on the project directory

**"Module not found" / "模块未找到"**: 
- 确保已安装所有依赖：`uv sync` 或 `pip install -r requirements.txt`
- Make sure all dependencies are installed

**端口 8080 被占用 / Port 8080 in use**: 
- 查找占用端口的进程：`lsof -i :8080`
- 或修改 Flask 应用的端口号
- Find process using port: `lsof -i :8080` or change Flask port

## 📚 故事创意示例 / First Story Ideas

- "一只害羞的机器人学习画杰作" / "A shy robot learning to paint masterpieces"
- "一位年轻科学家发现时间旅行" / "A young scientist discovering time travel"  
- "一只拯救社区的猫超级英雄" / "A cat superhero saving the neighborhood"
- "一只害怕飞行的龙" / "A dragon who's afraid of flying"
- "一个书籍会活过来的魔法图书馆" / "A magical library where books come alive"

## 📖 详细文档 / Detailed Documentation

更多详细信息，请查看：
For more detailed information, see:

- **本地测试指南** / **Local Testing Guide**: `本地测试指南.md`
- **完整 README** / **Full README**: `README.md`

**祝您使用愉快！/ Happy storytelling!** 🎨📚✨
