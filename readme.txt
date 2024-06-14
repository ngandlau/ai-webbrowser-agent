Todo
---
return prompts + answers in some kind of dataformat, e.g. csv or json. That data can act as input for some kind of debugger.
use openAI tool use or manually recreate the functionality? 
use recursion to call different agents?
proper system prompts vs user prompts
write a function to compare (input, output) pairs with several models, e.g. gpt-4o, gpt-3.5, gemini-1.5-flash, gemini-1.5-pro, ... => quickly test how different models perform the same task
hold everything fixed except 1 prompt?
understand how an LLM processes older messages vs previously written words.

Development Learnings
---
- hold certain responses fixed for end-to-end tests -- save money + faster
- test-driven development? fuzzy unit tests. LLM-unit test checks.

Links
---
Browser automation: https://github.com/Skyvern-AI/skyvern/
Vimium+GPT: https://github.com/ishan0102/vimGPT
Globot: https://github.com/Globe-Engineer/globot
AppAgent: https://appagent-official.github.io
Gumroad: https://www.gumloop.com/ -- web browser automation only in premium version ~$100/month
Microsoft AutoGen repository.
prompts: https://github.com/mnotgod96/AppAgent/blob/main/scripts/prompts.py