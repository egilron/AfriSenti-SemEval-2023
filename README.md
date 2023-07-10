# SemEval-2023 Task 12: Multilingual fine-tuning for sentiment classification in low-resource languages

Repository with our code used for fine-tuning and evaluation for [my contribution](https://arxiv.org/abs/2304.14189) towards the SemEval-2023 Shared Task 12.

The contribution evolved from a number of abandoned approaches. We have tried to share what was relevant for the final submission, and hope that the code is self-sufficient and readable. Please raise an issue if you find that something is missing or not in line with the reports in the paper.

## Datasets per language family / group
The training data per language category were prepared with [dataset_prep_2.ipynb](dataset_prep_2.ipynb) and are stored [here](datasets/train_group_final).  


## Links and citations
```
@misc{rønningstad2023uio,
      title={UIO at SemEval-2023 Task 12: Multilingual fine-tuning for sentiment classification in low-resource languages}, 
      author={Egil Rønningstad},
      year={2023},
      eprint={2304.14189},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@inproceedings{muhammadSemEval2023,
title = {{SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval)}},
author = {Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Seid Muhie Yimam and David Ifeoluwa Adelani and Ibrahim Sa'id Ahmad and Nedjma Ousidhoum and Abinew Ali Ayele and Saif M. Mohammad and Meriem Beloucif and Sebastian Ruder},
booktitle = {Proceedings of the 17th {{International Workshop}} on {{Semantic Evaluation}} ({{SemEval-2023}})},
publisher = {{Association for Computational Linguistics}},
year = {2023}
}

@misc{muhammad2023afrisenti,
title={{AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages}},
author={Shamsuddeen Hassan Muhammad and Idris Abdulmumin and Abinew Ali Ayele and Nedjma Ousidhoum and David Ifeoluwa Adelani and Seid Muhie Yimam and Ibrahim Sa'id Ahmad and Meriem Beloucif and Saif M. Mohammad and Sebastian Ruder and Oumaima Hourrane and Pavel Brazdil and Felermino Dário Mário António Ali and Davis David and Salomey Osei and Bello Shehu Bello and Falalu Ibrahim and Tajuddeen Gwadabe and Samuel Rutunda and Tadesse Belay and Wendimu Baye Messelle and Hailu Beshada Balcha and Sisay Adugna Chala and Hagos Tesfahun Gebremichael and Bernard Opoku and Steven Arthur},
year={2023},
doi={10.48550/arXiv.2302.08956},
url={https://arxiv.org/abs/2302.08956}
}
```


