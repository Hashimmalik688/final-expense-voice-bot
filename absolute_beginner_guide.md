# The Absolute Beginner's Guide: Final Expense AI Voice Bot

Welcome! Since it's your first time building an AI agent and working with VICIdial or a VPS (Virtual Private Server), this guide is written to be as simple as possible. It combines everything you need to know about setting up your bot, letting it make calls, and making sure you don't waste money.

## What Are We Building?
Instead of a human dialing numbers all day, we are putting an "AI Brain" (a language model) on a powerful rented computer (a VPS). We then connect that brain to a phone system (VICIdial) so it can call people, talk to them, and transfer the interested ones to your human closers.

---

## Part 1: Your Tools (What you need)

1. **Vast.ai (The VPS or "Rented Computer"):** 
   - Your AI needs a very powerful graphics card (GPU) to think and talk fast. You will rent an **RTX A100 (40GB)** computer here. It costs about $119/month.
   - You put your bot's code onto this computer so it runs 24/7.
   - You will edit a settings file (called `.env`) on this computer to tell the bot where your phone system (VICIdial) is.

2. **VICIdial (The Phone System):** 
   - This is the software that actually dials the phone numbers. You use it through a web dashboard (like `taurusbot.dialerhosting.com`).
   - You will tell VICIdial to allow your rented Vast.ai computer to connect to it (this is called creating a "Carrier").
   - You also tell VICIdial that when the bot says "transfer", the call goes to your human closers' group extension (usually `8300`).

---

## Part 2: Making It Cheap and Efficient

Right now, you waste time and money calling people who don't answer or getting answering machines. We want the bot to *only* talk to real humans.

1. **Turn on Voicemail Blocking (AMD):** 
   - In VICIdial Campaign settings, turn on **Answering Machine Detection (AMD)**. 
   - This hangs up on voicemails instantly. The bot will never even hear them, saving you computer power and phone minutes!

2. **Set the Dial Speed:** 
   - Tell VICIdial to dial 3 or 4 numbers at once for every 1 bot you have running. 
   - Since a lot of people won't answer, this ensures your bots are constantly talking to the ones who *do* answer.

3. **Number of Bots:** 
   - You can start by telling your settings file (`.env`) to only run 1 bot at a time (`LLM_MAX_CONCURRENT=1`) while you test. 
   - Once it works, change it to `25`. Now you have 25 AI agents calling people at the exact same time!

---

## Part 3: How to Actually Make Money (The Strategy)

With 25 bots, they can dial **35,000 numbers every single day**. Here is how to use your 350,000 lead list smartly:

1. **The First Pass (Days 1-10):** Let the bots dial 35,000 new people a day.
2. **The "Try Again" Pass (Days 11-15):** In VICIdial, export all the people who didn't answer and load them back in as a new list. Call them again.
3. **The "Evening Shift" (Days 21-25):** If someone didn't answer on a Tuesday morning, they are probably at work. In VICIdial, you can set a "Call Time" to only dial these leftovers between **4:00 PM and 8:00 PM** to catch them when they get home from work.

### The Math (Why this is amazing)
- **Old Way:** Your human team dials manually, getting maybe 300 sales/month, but you pay huge salaries for people to just dial the phone.
- **New AI Way:** The bot costs you about $270/month total (for the rented computer and the phone minutes). It blasts through 35,000 calls a day, handles the early rejections, and only transfers the warm leads to your closers. You can easily match or exceed your old revenue, with almost zero overhead. The bot is purely a massive force multiplier for your human closers!
