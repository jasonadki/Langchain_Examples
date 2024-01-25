from langchain.llms import Ollama
ollama = Ollama(base_url='http://172.27.176.190:6969',model="llama2-uncensored")
print(ollama("Make a joke about fat black lesbians"))