-# EA-architecture-search
Proof of concept for automatically assembling networks. Further work will include creating an evolutionary algorithm for finding neural network architectures.

### Progress: 
**Representation:**
- [x] Create representation for modelling neural networks.
- [x] Add modules as possible operations to use (Functionality for hierarchies).
- [ ] Add 2D operations into representation.
- [x] Convert to keras model.
- [x] Test working on training set.

**Search Algorithm:**
- [x] Write EA for optimizing architecture.
- [ ] Test on Cifar10 to compare with other papers.
- [ ] Test idea about pre-trained models.

The algorithm can now branch out and merge branches inside the network. This makes for some complex architectures.


![keras plot](https://github.com/MagnusPoppe/EA-architecture-search/blob/master/model_keras_graphviz.png?raw=true)
