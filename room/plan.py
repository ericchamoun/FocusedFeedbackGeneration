from room.step import Step


class Plan:

    steps = None
    current_step = None

    def __init__(self, primary_paragraph, instruction, text_steps) -> None:
        self.steps = [Step(t) for t in text_steps]
        self.current_step = 0
        self.primary_paragraph = primary_paragraph
        self.instruction = instruction


    def get_primary_paragraph(self):
        return self.primary_paragraph

    def has_next_step(self):
        return self.current_step < len(self.steps)

    def get_next_step_counter(self):
        return self.current_step + 1
    
    def get_next_step(self):
        return self.steps[self.current_step]
    
    def describe(self):
        return "\n".join([str(i+1) + ") " + step.describe() for i,step in enumerate(self.steps)])
    
    def describe_remaining_steps(self):
        return "\n".join([str(i+1) + ") " + step.describe() for i,step in enumerate(self.steps)][self.current_step:])
    
    def describe_next_step(self):
        desc = str(self.get_next_step_counter()) + ") " + self.get_next_step().describe()

        if not desc.endswith("."):
            desc += "."
        return desc

    def describe_resolutions_first_person(self):
        descs = []

        for i in range(0, self.current_step):
            r = self.steps[i].get_resolution()

            s = str(i + 1) + ") " + self.steps[i].describe()
            s += "\nYou asked the " + r["actor"] + " to " + r["action"] + ". It responded with:\n"
            s += r["description"]

            descs.append(s)

        return "\n\n".join(descs)
    
    def describe_resolutions_third_person(self):
        descs = []

        for i in range(0, self.current_step):
            r = self.steps[i].get_resolution()

            s = str(i + 1) + ") " + self.steps[i].describe()
            s += "\nThe controller asked the " + r["actor"] + " to " + r["action"] + ". It responded with:\n"
            s += r["description"]

            descs.append(s)

        return "\n\n".join(descs)
    
    def get_steps(self):
        return self.steps

    def update(self, plan_resolution):
        self.steps[self.get_next_step_counter():self.get_next_step_counter()] = plan_resolution["plan"].get_steps()

    def resolve_current_step(self, resolution):
        self.get_next_step().resolve(resolution)
        self.current_step += 1

    def get_question_answer_evidence(self):
        descs = []

        for i in range(0, self.current_step):
            r = self.steps[i].get_resolution()

            # Skip anything not produced by the searcher
            try:
                if r["actor"] != "Investigator":
                    continue
                if len(r["answers"]) == 0:
                    continue
                s = r["question"] + "\n"
                for answer in r["answers"]:
                    s += " * According to " + answer["source"] + ", \""+ answer["backing"] + "\"\n"

                descs.append(s)
            except:
                continue

        return "\n".join(descs)