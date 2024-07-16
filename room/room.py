from agents.controller import Controller
from agents.planner import Planner
from agents.reviewer import Reviewer
from agents.search_agent import SearchAgent
from room.plan import Plan
import warnings
warnings.filterwarnings("ignore")

class Room:

    plan = None
    agent_dict = None
    logger = None

    def __init__(self, logger) -> None:
        self.controller = Controller(logger)
        self.search = SearchAgent()
        self.planner = Planner()
        self.reviewer = Reviewer()
        self.logger = logger

        self.agent_dict = {
            "controller": self.controller,
            "planner": self.planner,
            "investigator": self.search,
            "reviewer": self.reviewer,
        }

        action_options = []
        super_action_options = []
        agents = [self.search, self.reviewer]
        super_agents = [self.controller, self.planner]
        for agent in agents:
            #action_options.extend(agent.get_action_shorthands())
            action_options.extend(agent.get_actions())
            super_action_options.extend(agent.get_actions())

        for agent in super_agents:
            super_action_options.extend(agent.get_actions())

        self.planner.set_options(
            action_options
        )
        
        self.controller.set_options(
            super_action_options
        )

        self.plan = None

    def get_markdown_summary(self):
        markdown = "# Overall goals\n\n"
        markdown += "As a scientific reviewing model, write the weaknesses and areas of improvement of the following passage" + self.plan.get_primary_paragraph() + "\"\n"
        markdown += "# Session log\n\n"
        markdown += self.logger.history()

        return markdown

    def greet(self):
        return "Hello! I'm ReviewGPT, your friendly scientific reviewing AI. Please tell me what I should review."
    
    def format_action(self, action):
        act_str = "[" + action["actor"] + "]: " + action["action"]
        if "parameters" in action and len(action["parameters"]) > 0:
            act_str += " (" + ", ".join([k+"="+str(v) for k,v in action["parameters"].items()]) + ")"

        return act_str
    
    def execute_action(self, action):
        if action["actor"].lower() not in self.agent_dict:
            return {
                "actor": "System",
                "description": "There is no agent called \"" + action["actor"] + "\"."
            }
        else:
            agent = self.agent_dict[action["actor"].lower()]

            resolution = agent.interpret(action)

        # If the plan has changed, update it:
        if agent == self.planner:
            self.plan.update(resolution)
            self.logger.log("Controller", "I have adopted the new plan.")

        return resolution

    def main_loop(self, user_message, paper_url, instruction):
        # We always start by making a plan:
        self.plan = Plan(user_message, instruction, [])
        self.controller.set_plan(self.plan)
        action = {
            "actor": "Planner",
            "action": "Make a scientific reviewing plan",
            "parameters": {
                "paragraph": user_message,
                "instruction": instruction
            }
        }
        current_step = 0
        action["paragraph"] = user_message
        action["instruction"] = instruction
        action["plan"] = self.plan
        action["current_step"] = current_step
        self.execute_action(action)


        while self.plan.has_next_step():
            current_step += 1
            action = self.controller.choose_next_action()
            action["paragraph"] = user_message
            action["plan"] = self.plan
            action["paper_url"] = paper_url
            action["instruction"] = instruction
            action["current_step"] = current_step
            resolution = self.execute_action(action)
            
            if self.plan.has_next_step():

                self.plan.resolve_current_step(resolution)
            
            else:

                current_step += 1

                # We are done with the plan. Try evaluating the paragraph three times:
                action = {
                "actor": "Reviewer",
                "action": "Write review",
                }
                action["paragraph"] = user_message
                action["instruction"] = instruction
                action["plan"] = self.plan
                action["current_step"] = current_step
                print("[CONTEXT]")
                print(self.plan.get_question_answer_evidence())
                print("[REVIEW]")
                resolution = self.execute_action(action)

                try:
                    verdict = resolution["review"]
                except:
                    return 1

        return resolution
