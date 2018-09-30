# Evolutionary algorithm for Neural Architecture Search
Proof of concept for automatically assembling networks. This application creates several neural network models and evolves them to find the optimal architecture for any given classification task. 

### Genotype: 
The genotype selected for this project is an acyclic directed graph containing either layers/operations or other acyclic directed graphs of the same type. This is then a hierarchy of operations ordered by the graph. 


![Genotype illustration](https://github.com/MagnusPoppe/EA-architecture-search/blob/master/model_images/genotype.png?raw=true)


The Genotype is decoded into a phenotype (keras model) using a breadth first based algorithm. 

**Work progress can be followed from the trello board:**

https://trello.com/b/oQ66wUbx
