class Step:

    description = None

    assigned_agent = None
    assigned_action = None

    resolution = None

    def __init__(self, description) -> None:
        self.description = description

    def assign(self, agent, action):
        self.assigned_agent = agent
        self.assigned_action = action

    def resolve(self, resolution):
        self.resolution = resolution

    def describe_resolution(self):
        return self.resolution["description"]
    
    def describe(self):
        return self.description

    def get_resolution(self):
        return self.resolution