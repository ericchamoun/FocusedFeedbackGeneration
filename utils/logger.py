class Logger:

    __log__ = None

    def __init__(self) -> None:
        self.__log__ = []

    def log(self, actor, message):
        self.__log__.append([actor, message])
        print("[" + actor + "]: " + message)

    def history(self):
        return "\n".join(["[" + actor + "]: " + message for actor, message in self.__log__])