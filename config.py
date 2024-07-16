llm = "gpt-4"
embedding_model = "text-embedding-3-large"
openai_key = ""
google_api_key = ""
search_engine_id = ""
instruction = """ EXAMPLES: \n
               Paragraph : \"The reduced incidence rate of recent years has been attributed to better equipment and an emphasis on personal protective measures (Rowland 2015). [NEXT] Nevertheless, there is still a cause for concern when troops are newly deployed to endemic regions; supportive resources may not be fully in place, deployed personnel may have limited knowledge, and a culture of preventative measure necessity may not have yet developed (Coleman 2006).\"
               {\"label\": \"Meaningful Comparison\", \"review\": \"I suggest that the authors can cite more current US military infection data, since the authors cite "Rowland 2015" as a reference in the passage "The reduced incidence rate of recent years has been attributed to better equipment and an emphasis on personal protective measures". What is this most current incidence rate?\", \"reasoning\": \"This paper is a a review of malaria and leishmaniasis among United States Armed Forces that was published in 2019. In this paragraph, the authors use the study published by Rowland in 2015 to claim that there was a reduction in the incidence rate of leishmaniasis in recent years and explain the reason for it. However, as the paper is published 4 years later the study from Rowland, it is essential that the author cites a more recent study that provides more recent figures of the incidence rate of leishmaniasis among United States Armed Forces for the comparison to be meaningful.\" } \n
               
               Paragraph: \"As expected, security in microservice systems gained a lot of academic interest in the latest years. [PREVIOUS] This is reflected by the sharp increase in the number of publications since 2014.\"
               {\"label\": \"Empirical and Theoretical Soundness\" , \"review\": \"This is reflected by the sharp increase in the number of publications since 2014." I am not convinced by this argument. The overall number of publications in computers science is steadily increasing, so the absolute number of papers does not say much.\", \"reasoning\": \"The paper presents a systematic review of the state of the art of microservice security. The authors construct and analyze a dataset consisting of 290 peer-reviewed publications and, among others, perform a quantitative analysis on the metadata of the publications. In this paragraph, the authors highlight that the number of publications in the field of microservice security has experienced a sharp increase since 2014 and infer from that the academic interest in the field has substantially increased. However, as the overall number of publications in computer science has also steadily increasing over the years, the increase of publications in a specific field does not necessarily show an increase in interest in this particular field. Therefore, the authors make an incorrect conclusion from the observed data.\" } \n
               
               Paragraph: \"Shortly after the COVID-19 outbreak various machine learning algorithms have been implemented [10]- [13]. [PREVIOUS] Machine learning helps to quickly identify patterns and trends of the large volume of data, that are difficult for humans to recognize [14]. [NEXT] The availability of objective stratification tools to rapidly assess a patient status and prognosis is of a great use for the frontline health-care providers [15]. \"
               {\"label\": \"Substance\" , \"review\": \"Although the authors correctly state that " Machine learning helps to quickly identify patterns and trends of the large volume of data, that are difficult for humans to recognize". However it seems that the number of variables and observations in the study are small enough for statistical models or the more interpretable machine learning approaches like lasso, ridge regression or other penalized approaches.\", \"reasoning\": \"The study aims to assess gastrointestinal and liver-related predictive factors for SARS-CoV-2 associated risk of hospitalization. The authors propose to use machine learning algorithms as these allow to identify patterns in large volume of data. However, in this case, only 710 patients were enrolled in the study, which is not a large volume of data. For this reason, to make the study more complete, the reviewer is suggesting more experiments using statistical models or more interpretable machine learning approaches such as lasso or ridge regression.\"} \n
               
               Paragraph: \"2016), this paper proposes a framework to identify event factuality in raw texts with neural networks. [PREVIOUS] Our main contributions can be summarized as follows: 1) the proposal of a two-step supervised framework for identifying event factuality in raw texts. [NEXT]  2) the utilization of an attention-based CNN to detect source introducing predicates (SIPs), the most important factors to identify event factuality.\"
               {\"label\": \"Originality\" , \"review\": \"While I feel that the work is original in engineering deep neural nets for the factuality classification task, and that such work is valuable, its approach is not particularly novel, and "the proposal of a two-step supervised framework" is not particularly interesting given that FactBank was always described in terms of two facets \", \"reasoning\": \"The study proposes a deep learning framework for event factuality which first extracts essential information from raw texts as the inputs and then identifies the factuality of events via a deep neural network with a proper combination of Bidirectional Long Short-Term Memory (BiLSTM) neural network and Convolutional Neural Network (CNN). The reviewer gives credit for the originality in using deep neural nets for the factuality classification task. However, they explain that FactBank corpus (Sauri and Pustejovsky, 2009), on which the experiments are performed is already described in terms of two facets. Therefore, the claim that “the proposal of a two-step supervised framework” is a novel contribution by the author of the paper is not accurate given that it was based on an idea from previous work in the area.\"}\n
               
               Paragraph: \"To train our BP with reinforcement learning, we base our implementation on the open-source code of Other-Play (Hu et al., 2020) , which includes several recent advancements for RL in Hanabi. We follow most of their practices such as value decomposition network, color shuffling and auxiliary tasks. We also leave their hyper-parameters unchanged. \"
               {\"label\": \"Replicability\" , \"review\": \"The experimental setup requires significantly more details on the hardware used for training , testing and validating .\", \"reasoning\": \"The paper presents Learned Belief Search (LBS): a computationally efficient search procedure for partially observable environments. In the experimental setup, while discussing the BluePrint training, the authors simply state that they “base their implementation on the open-source code of Other-Play (Hu et al., 2020)”. However, it is crucial that the authors provide more detail of their training such as the hardware used for training, testing and validating in order for a reader to be able to replicate the experiments.\"}\n
               """
