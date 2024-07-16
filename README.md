# Automated Focused Feedback Generation for Scientific Writing Assistance

In this work, we propose the task of focused feedback generation for scientific writing assistance. Focused feedback entails providing specific, actionable, and coherent comments that identify weaknesses in a paper and/or suggest revisions. 

To this end, we present SWIF<sup>2</sup>T: a Scientific WrIting Focused Feedback Tool. Our approach consists of four components - planner, investigator, reviewer and controller - leveraging multiple Large Language Models (LLMs) to implement them. We compile a dataset of 300 peer reviews citing weaknesses in scientific papers and conduct human evaluation. The results demonstrate the superiority in specificity, reading comprehension, and overall helpfulness of SWIF<sup>2</sup>T’s feedback compared to other approaches.

[Automated Focused Feedback Generation for Scientific Writing Assistance](https://arxiv.org/pdf/2405.20477). Eric Chamoun, Michael Schlichktrull, and Andreas Vlachos (ACL Findings 2024).

**Running SWIF<sup>2</sup>T**

1. Setup conda environment

```
conda create -n "focused_feedback_generation" python=3.12.1
conda activate focused_feedback_generation
```

2. Download the reranking model into the *plan_reranking_inference/* directory from: https://drive.google.com/file/d/1wRrown4YhKN3PiUdTm689sV-w9wiCdBC/view?usp=sharing
  
3. Update preferences for models and write API keys in config.py

4. Add the paper you would like to get reviewed in the root directory

5. Run the model

```
python run.py
```

**Future work**

Future work will focus on enhancing literature-augmented question answering, implementing a weakness detection model and using recent open-source LLMs to decrease the costs associated with running SWIF<sup>2</sup>T. If you are interested in collaborating, please do not hesitate to reach out.

**Citation**

If you find this useful, please cite our paper as:

    @misc{chamoun2024automated,
        title={Automated Focused Feedback Generation for Scientific Writing Assistance}, 
        author={Eric Chamoun and Michael Schlichktrull and Andreas Vlachos},
        year={2024},
        eprint={2405.20477},
        archivePrefix={arXiv},
        primaryClass={cs.CL}
    }
