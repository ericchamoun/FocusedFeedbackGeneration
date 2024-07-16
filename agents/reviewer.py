from agents.agent import Agent


class Reviewer(Agent):

    options = None
    system_message = "You are the ReviewGPT Reviewer, a world class AI assistant for scientific reviewers. You write a review that highlights the weaknesses and areas of improvements of a paragraph based on context given to you, and return results as valid JSON. You need to make sure that the review addresses a specific portion of the paragraph, that it is not generic and that it is constructive. If you think that no review is needed, then you can also say this. Also, make sure to use the given context to generate a review: a review that points out the absence of an information in the paragraph should not be made if this information is present in another paragraph of the paper, be careful this is very important!"

    def set_options(self, options):
        self.options = options
        
    # Potential extension: creating a Critic LLM to verify the review.        
    def write_review(self, paragraph, plan, instruction):

        prompt = "You will be given a paragraph with the following context:\n\n"

        try:
            prompt += plan.get_question_answer_evidence()
        except:
            print("No context found")

        prompt += """There are five possible review labels: Empirical and Theoretical Soundness, Meaningful Comparison, Substance, Originality, Replicability. Write a review that:
                     1- Selects and quotes a substring from the given paragraph.
                     2- Chooses the appropriate review label 
                     3- Writes a review sentence that points out a weakness or suggests an improvement (if needed) using the quoted substring, the review label and the context. It is IMPORTANT that you use the provided context to generate a sensible review.  
                     4- Generates JSON object with the keys \"reasoning\",\"review\" and \"label\". Below are examples that follow all these rules, use them as inspiration: \n """ + instruction

        prompt += "Paragraph: " + paragraph

        messages=[
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        result_message = self.send_message(messages)

        print(result_message)
        
        return 1
   
    def get_actions(self):
        return [
            "Actor: Reviewer | Action: Write review | Parameters: | Description: Write a review that only points out the weaknesses and areas of improvement of a passage based on the plan so far. Can only be called once context has been gathered by another agent."
        ]
    
    def interpret(self, action):
        if action["action"] == "Write review":
            return self.write_review(action["paragraph"], action["plan"], action["instruction"])
        else:
            return {
                "actor": "Reviewer",
                "description": "This action is not in my set of possible instructions.",
                "action": action["action"]
            }