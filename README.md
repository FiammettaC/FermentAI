# FermentAI

## Abstract
Recent developments in artificial intelligence (AI), leading to the release of continuously improving large language models (LLMs) provide the opportunity for educators to automate repetitive tasks with stimulant experiences for the students. This ability of LLMs to extract content and key information from text offers a powerful tool for enhancing the learning experience. In this work, we present an example of how LLMs can be used to automate educational processes. We implement FermentAI, a virtual tutor (VT) to answer students’ questions about fermentation for a Master’s Degree course taught at the technical University of Denmark. The model used is a pre-trained sequence-to-sequence model. The prompt provided to the LLM is composed of a question (asked by the student) and its context, containing the curated information for the model to answer the question. The context is retrieved through a semantic search by calculating the cosine similarity between the query question and the most similar historical question. The primary objective of this work is to create an interactive tool, which is freely available online and returns accurate responses, which students can use to ask questions and clarify any doubts regarding fermentation. Additionally, this work aims to improve students’ learning experience through stimulating material and gamification. 

## Contacts
If you have any questions, feel free to contact Fiammetta Caccavale (fiacac@kt.dtu.dk).

## Cite this work
If you would like to cite this work, please reference our paper: **SPyCE: A structured and tailored series of Python courses for (bio)chemical engineers**.
```bibtex
@incollection{caccavale2024fermentai,
  title={FermentAI: Large Language Models in Chemical Engineering Education for Learning Fermentation Processes},
  author={Caccavale, Fiammetta and Gargalo, Carina L and Gernaey, Krist V and Kr{\"u}hne, Ulrich},
  booktitle={Computer Aided Chemical Engineering},
  volume={53},
  pages={3493--3498},
  year={2024},
  publisher={Elsevier}
}
```
