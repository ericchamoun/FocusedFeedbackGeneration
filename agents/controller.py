from email import utils
from agents.agent import Agent
import json

class Controller(Agent):

    options = None    
    project = None
    engineer = None
    plan = None
    current_plan_step = None
    logger = None

    system_message = "You are the ReviewGPT Controller, a helpful scientific reviewing assistant. You manage several other AIs, passing directions to them from the user. You communicate directly with the other AIs, and as such your answers MUST be ONLY valid json."
    
    def __init__(self, logger) -> None:
        super().__init__()
        self.options = []
        self.current_plan_step = 0
        self.logger = logger

    def get_actions(self):
        return [
            "Actor: Controller | Action: Skip this step | Parameters: | Description: Skip the current step if it is unnecessary or impossible"
        ]
    
    def set_plan(self, plan):
        self.plan = plan
    
    def interpret(self, action):
        if action["action"] == "Start next plan step":
            return self.execute_next_step()
        elif action["action"] == "Skip this step":
            return {
                "actor": "Controller",
                "action": "Skip this step",
                "description": "I skipped this step."
            }

    
    def set_options(self, options):
        self.options = options
    
    def choose_next_action(self):
        prompt = "You are currently following an overall plan to point out the weaknesses in the paragraph \""+self.plan.get_primary_paragraph()+"\"."

        if self.plan.get_next_step_counter() > 1:
            prompt += " This is a log of your progress so far:\n\n"
            prompt += self.plan.describe_resolutions_first_person()
        else:
            prompt += ""

        prompt += "\n\nThe remaining steps are:\n\n"
        prompt += self.plan.describe_remaining_steps()
        prompt += "\n\nThe next step is " + self.plan.describe_next_step()

        prompt += "\n\nYou will be given a list of actions. Your task is to decide what the best action to take is to accomplish the next step. Each action has several fields, separated by a vertical line (|). These are the actor who takes the action, the name of the action, the parameters that action requires, and a short description of the action. The options are:\n"
        prompt += "\n".join([" * " + o for o in self.options]) + "\n\n"

        prompt += "Provide the best action to take. Your answer must be valid JSON. It should be a JSON object with four entries, \"explanation\", \"actor\", \"action\", \"parameters\". Actors and actions should be strings, parameters should be another JSON object. Explanations should be a string containing a step-by-step description of why you chose this action. Remember, necessary parameters can be found between curly brackets in the commands. Output JUST the command, using your imagination if something is missing."
        
        messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        result = "{" + "}".join("{".join(self.send_message(messages).split("{")[1:]).split("}")[:-1]) + "}"
        action = json.loads(result)

        self.logger.log("Controller", "I am executing step " + self.plan.describe_next_step() + " " + action["explanation"].strip())

        act_str = action["action"]
        if "parameters" in action and len(action["parameters"]) > 0:
            act_str += " (" + ", ".join([k+"="+str(v) for k,v in action["parameters"].items()]) + ")"

        self.logger.log("Controller", "I am asking the " + action["actor"] + " to \"" + act_str + "\"")

        return action

    def parse(self, message):
        prompt = ""

        if len(self.plan) > 0:
            prompt += "Please note: You are currently following an overall plan. The step you are currently working on is " + str(1+self.current_plan_step) + " (" + self.plan[self.current_plan_step]+"). The full plan is:\n"
            prompt += "\n".join([str(1+i) + ") " + x for i,x in enumerate(self.plan)])
        else:
            prompt = "Warning: You are not currently following any plan. I suggest making one."

        prompt += "\n\nYou will be given a list of actions, and a message. Your task is to decide, given the message, what the best action to take is. Each action has several fields, separated by a vertical line (|). These are the actor who takes the action, the name of the action, the parameters that action requires, and a short description of the action. The options are:\n"
        prompt += "\n".join([" * " + o for o in self.options]) + "\n\n"
        prompt += "The message is:\n"
        prompt += message + "\n\n"
        prompt += "Provide the best action to take. Your answer must be valid JSON. It should be a JSON object with four entries, \"explanation\", \"actor\", \"action\", \"parameters\". Actors and actions should be strings, parameters should be another JSON object. Explanations should be a string containing a step-by-step description of why you chose this action. Remember, necessary parameters can be found between curly brackets in the commands. Output JUST the command, using your imagination if something is missing."
        messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        result = "{" + "}".join("{".join(self.send_message(messages).split("{")[1:]).split("}")[:-1]) + "}"
        result = json.loads(result)
        result["message"] = message
        
        return result