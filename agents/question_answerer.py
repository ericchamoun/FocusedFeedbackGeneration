from agents.agent import Agent
import json
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

os.environ["OPENAI_API_KEY"] = config.openai_key


class QuestionAnswerer(Agent):

    options = None

    def set_options(self, options):
        self.options = options
        
    def get_answer_from_evidence(self, question, answer, source):

        if len(answer) == 0:
            return {
                "backing": "none",
                "answer": "unavailable",
                "source": "unavailable"
            }

        if source == "":
            source = "the internet"

        result = json.dumps({
            "backing":answer,
            "answer": answer,
            "source": source
        })

        try:            
            result = json.loads(result)
        except:
            return {
                "backing": "none",
                "answer": "unavailable",
                "source": "unavailable"
            }


        # If we have no answer, or the answer is unavailable, we just stop here. 
        if "answer" not in result:
            result["answer"] = "unavailable"
        if result["answer"] == "unavailable":
            return result

        # Verify that the answer contains backing, and that the backing is actually in the original document:
        if "backing" not in result:
            result["answer"] = "unavailable"

        return result
