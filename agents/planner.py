from agents.agent import Agent
from room.plan import Plan
from transformers import AutoTokenizer
import logging
import torch
from plan_reranking_inference import prediction, dataset, planner_reranking
import numpy as np
logging.getLogger("transformers").setLevel(logging.ERROR)

class Planner(Agent):

    options = None
    system_message = """You are the ReviewGPT Planner, a world class scientific reviewing assistant. You create plans using the Investigator and Reviewer AI agents to review paragraphs. You will ask the Investigator to gather context from both the web and the paper in the first few steps, then, at the end, the action you ask the Reviewer agent will be exactly “Write a review based on the gathered context.”. DO NOT add a single word to this sentence. Your output MUST be formatted as a numbered list. NEVER write a step that does not involve an action for the Investigator or the Reviewer agents"""
            
    def __init__(self) -> None:
        super().__init__()

    def set_options(self, options):
        self.options = options
        
    def make_plan(self, paragraph, instruction, number_plans_to_rerank = 4):

        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        model = planner_reranking.RerankingModel.from_pretrained("bert-base-cased")
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load("plan_reranking_inference/reranking_model.pt", map_location='cpu'))

        prompt = "You will be given a paragraph. Your task is to point out the weaknesses of this paragraph, i.e. ask questions to gather context from the paper and the web before reasoning over it and the paragraph to identify the weaknesses of the passage. Thinking step by step, break the process of scientific reviewing down into small, simple tasks. These should involve gathering context for the paragraph, i.e., gathering information from the paper (such that the paragraph can be understood, verified and criticized without requiring any access to the paper.) and from the literature (such that the Reviewer AI agent can understand cited studies, compare the paper againt other related studies, evaluate its originality and soundness and be aware of criticisms and limitations, all without needing any access to the literature. The questions should be self-contained and formulated to facilitate effective Google searches). The gathered information should allow the Reviewer to comment on the soundness, originality, replicability, meaningfulness of the comparison or the substance of the information discussed in the paragraph. As you make plans for other AI tools, each step should be solvable using one of the following actions:\n"
        prompt += "\n".join([" * " + o for o in self.options]) + "\n\n"
        prompt += """Your plan should be a numbered list. Steps should be in simple language, and mention which agent should do them. I will give you now the golden rules by which you NEED to abide. It is of upmost important that none of these rules is broken:
                     Rule #1: Each step involves requesting the Investigator or the Reviewer to perform an action.
                     Rule #2: The plan begins with the Investigator answering questions using the paper. Each of these steps should start with \" Search the paper to understand \. The next steps should request the Investigator to answer questions by searching the web. These should start with \" Search the web to understand \" and should be only about one idea. Questions answerable from the web should be self-contained such that they are understandable without referring to another step or the paper. Finally, the last step should be EXACTLY \"Reviewer: Write a review based on the gathered context, which should point out the weaknesses of the paragraph.\"
                     Rule #3: The questions to the Investigator should ONLY ask about one concept at a time. For example, \" Search the paper to understand what Attention is \" is valid but \" Search the paper to understand what Attention is and how it works \" is NOT. 
                     IT IS IMPORTANT TO RESPECT THE THREE GOLDEN RULES I JUST GAVE YOU. Now, the paragraph you will review is: \""""
        prompt += paragraph + "\"."

        messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        data_for_reranking = []
        plans = []
        for _ in range(number_plans_to_rerank):
            result = self.send_message(messages)
            plans.append(result)
            data_for_reranking.append((paragraph,result,result)) #In format of pairwise re-ranker
        
        dataloader = dataset.DataLoader(data_for_reranking,tokenizer, 1,512)
        dataloader = dataloader._get_data_loader()
        prediction_obj = prediction.Prediction(model, dataloader)
        logits = prediction_obj.predict()[1]
        best_plan_index = np.argmax(logits)
        best_plan = plans[best_plan_index]
        print(best_plan)
                
        steps = [" ".join(x.strip().split(" ")[1:]) for x in best_plan.split("\n")]
        steps = [step for step in steps if step]

        plan = Plan(paragraph, instruction, steps)

        resolution = {
            "actor": "Planner",
            "description": "I have made the following plan:\n\n"+plan.describe(),
            "plan": plan,
            "action": "make a scientific reviewing plan"
            }
        
        return resolution

   
    def get_actions(self):
        return [
        ]
    
    def interpret(self, action):
        if action["action"] == "Make a scientific reviewing plan":
            return self.make_plan(action["parameters"]["paragraph"], action["parameters"]["instruction"])