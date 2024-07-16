import openai
from time import sleep
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

class Agent:

    prompt_store = None

    def __init__(self) -> None:
        self.model = config.llm

    def send_message(self, message):
        worked = 0
        sleep_time = 1
        while worked == 0:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=message
                )
                text = response.choices[0].message.content
                worked = 1
            except:
                print("\nRetrying LLM query with sleep time: " + str(sleep_time))
                sleep(sleep_time)
                sleep_time += 1

        return text
    
    def get_action_shorthands(self):
        sh = []
        for action in self.get_actions():
            actor = action.split("|")[0].strip()
            description = action.split("|")[3].strip()

            sh.append(actor + " , " + description)

        return sh

    def parse(self, text):
        return self.send_message(self.prompt_store.get_project_init_prompt(text))