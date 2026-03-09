# Final Expense Voice Bot - Absolute Beginner Copy/Paste Guide

This guide contains the exact commands you need to run your AI bot on Vast.ai. 

## Step 1: Rent Your Server
1. Go to Vast.ai and rent an **RTX A100 (40GB)** PCIe instance.
2. **Important:** When picking the OS Image, use the `nvidia/cuda:12.1.1-devel-ubuntu22.04` template.

## Step 2: Install the Bot
Once your Vast.ai server is running, open the terminal (Click "Connect" -> open the SSH or Jupyter terminal) and paste this exact block of code:

```bash
git clone <your-repo-link-here> voicebot
cd voicebot
pip install -r requirements.txt
cp .env.example .env
```

## Step 3: Add Your Settings
You need to tell the bot where your phone system is. Open the `.env` settings file by pasting this:

```bash
nano .env
```

Now, fill in your actual VICIdial information and SIP passwords. Put your Vast.ai machine's Public IP address in the `SIP_LOCAL_IP` field.

*To save and exit the text editor: Press `Ctrl+X` on your keyboard, then press `Y`, then press `Enter`.*

## Step 4: Start the AI
Paste this final command to turn the bot on:

```bash
./scripts/deploy_gpu.sh
```

*(Note: The very first time you run this, it will take ~10 minutes to download the AI "Brain". Do not close the window while it downloads!)*

### That's it! 
Your bot is now running and waiting for VICIdial to send it phone calls. Read the `absolute_beginner_guide.md` file if you need help with the VICIdial side of the setup.
