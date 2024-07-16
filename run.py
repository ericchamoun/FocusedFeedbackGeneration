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
#paragraph = """\u2022 We propose a general framework of FL with partial model personalization, which relies on domain knowledge to select a small portion of the model to personalize, thus imposing a much smaller memory footprint on the devices than full model personalization. This framework unifies existing work on personalized FL and allows arbitrary partitioning of deep learning models. \u2022 We provide convergence guarantees for the FedSim and FedAlt methods in the general (smooth) nonconvex setting. While both methods have appeared in the literature previously, they are either used without convergence analysis or only analyzed for convex models. Our analysis in the nonconvex setting provides the first theoretical support of them for training general deep learning models. The analysis of FedAlt with partial participation is especially challenging and we develop a novel technique of virtual full participation to overcome the difficulties. to the identity mapping if vi = 0; in addition, it has a bottleneck in the middle (Houlsby et al., 2019) . (b) The generalized additive model can be further augmented with a shared input layer for representation learning.\n\u2022 We conduct extensive experiments on image classification and text prediction tasks, exploring different model personalization strategies for each task, and comparing with several strong baselines. Our results demonstrate that partial model personalization can obtain most of the benefit of full model personalization with a small fraction of personalized parameters, and FedAlt often outperforms FedSim. \u2022 Our experiments also reveal that personalization (full or partial) may lead to worse performance for some devices, despite improving the average. Typical forms of regularization such as weight decay and dropout do not mitigate this issue. This phenomenon has been overlooked in previous work and calls for future research to improve both performance and fairness."""
#paper_link = "/Users/ericchamoun/Desktop/Work/Focused Feedback Generation/ReviewGPT-3.0/run.py"

result = room.main_loop(paragraph, paper_link, config.instruction)
print(result)
