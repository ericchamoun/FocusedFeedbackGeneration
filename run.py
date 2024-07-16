import pandas as pd
from tqdm import tqdm
import openai
from room.room import Room
from utils.logger import Logger
import config


openai.api_key = config.openai_key


logger = Logger()
room = Room(logger)
logger.log("Controller", room.greet())
print("Enter the paragraph you would like to have reviewed:")
paragraph = input(" > ")
print("Enter the link of the paper:")
paper_link = input(" > ")

result = room.main_loop(paragraph, paper_link, config.instruction)
