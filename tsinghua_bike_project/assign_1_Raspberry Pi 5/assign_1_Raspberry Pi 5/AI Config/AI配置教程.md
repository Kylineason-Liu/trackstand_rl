# AI配置教程

在编程的过程中，如果没有AI的话，需要在各种网站上搜索各种库的资料，然后边查看资料边编写代码；其中检索资料是及其消耗时间和精力的。而且即便找到了资料，一般来说，资料也是及其繁杂的，你想要的东西和不想要的东西都是混杂在一起的，总而言之就是效率非常低。

所以我推荐大家多使用AI进行学习和Coding，但切记我们需要审阅、理解AI所生产的东西。如果有任何地方不理解，都要和AI 进行battle，一定要完全理解。

我目前使用得比较多的AI 工具是CodeX和Claude Code。不过大陆这边他们原厂的AI模型被封锁了，所以大家可以搭配国产的DeepSeek的API(DeepSeek便宜)。

我们希望AI能够直接帮我们在树莓派上工作，所以需要AI的工具部署到树莓派上。以下的命令全在树莓派的终端里运行哟~

## Claude Code + DeepSeek

具体该怎么做呢？其实总体思路也非常清晰：

1. 购买DeepSeek API
2. 下载Claude Code
3. 配置Claude Code使用DeepSeek

注册DeepSeek需要大陆的手机号，如果没有的话可以使用助教的DeepSeek API。如果有大陆的手机号，推荐大家跟着教程体验一下API的购买和使用过程。https://zhuanlan.zhihu.com/p/2032198932500059424。

获取到API Key之后可以先以txt文件的形式存在自己的电脑上。我们先下载Claude Code。

在下载Claude Code之前，我们需要先安装一下它的语言环境Node.js

先把nvm_install.sh传到树莓派上(可以打开VSCode连接之后，直接拖到Explorer里),然后在树莓派里运行这个脚本
```bash
bash nvm_install.sh
```

更新一下环境变量，让nvm能被找到
```bash
source ~/.bashrc
```

通过nvm安装nodejs的最新稳定版本
```bash
nvm install node
```

安装好了之后就能安装Claude Code了
```bash
npm install -g @anthropic-ai/claude-code
```

运行这个命令，如果显示版本号就说明安装成功了
```bash
claude --version
```

然后则是需要配置Claude使用DeepSeek，我们使用VSCode连接树莓派，打开~/home/ubuntu~/.claude/settings.json把它改成这样
```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "sk-900384bc26344169b6cd7d6fd044bf2b",
    "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash[1m]"
  }

}
```

保存之后，在终端运行claude就能够使用了

## 结合VSCode使用Claude Code
让

通过Remote Explorer连接上树莓派了之后，在Extensions 的 Extension Marketplace里给树莓派下载并安装Claude Code的插件

下载完毕后，可以在右上角看到Claude Code的图标，点击之后，会弹出来一个侧边窗口，等待若干秒，它就能读取完毕配置，从而能够正常使用了。