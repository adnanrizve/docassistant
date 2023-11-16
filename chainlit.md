These simple bots will allow you to chat with PDF files. The local LLM which these bots are going to use by default is https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q8_0.gguf. I am using "BAAI/bge-small-en-v1.5" as embedding model due to its high score on leaderboard https://huggingface.co/spaces/mteb/leaderboard

This repo have two apps (Chainlit based), you can run "allinone.py" through chainlit command if you would like to use GUI to upload the file and then chat with a document. The other app is "answerbot.py"  which allow you ask question regarding multiple documents but you need to use "ingest.py" to perform the embedding process first.

Special thanks to https://github.com/AIAnytime/Llama2-Medical-Chatbot and https://docs.chainlit.io/examples/qa

Let's Chat!
