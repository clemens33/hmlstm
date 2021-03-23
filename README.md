## HMLSTM

- my reimplementation of the paper

<pre>
J. Chung, S. Ahn, and Y. Bengio, “Hierarchical multiscale recurrent neural networks,” arXiv preprint arXiv:1609.01704, 2016.
</pre>

**disclaimer**
- work was done during a seminar/project in bioinformatics+ai master program
- I don't claim that my results/findings/implementations in context to the HMLSTM architecture are correct

### Content
- [documents](./documents) - contains all my written work (seminar paper + presentation slides + results)
- [hmlstm](./hmlstm) - contains the implementation - if you just want to use it focus on "network.py" and "HMLSTMNetwork" - class.
- [lstm](./lstm) - can be ignored - was only used as a baseline for comparison  
- [projects](.projects) - contains the dataset - implementation was used/tested on a character modeling task (predict next character)
- [environment.yml](./environment.yml) - contains the conda env

### Results/Architecture
- The architecture is very interesting - if you want to learn about it focus on the seminar paper in the documents' folder - I spent quite a while on visualizations
- It is basically a stacked LSTM which learns to mask out information when information is going from bottom to top stacked LSTMs.
- This mask/boundary detector can be used for visualization (which boundaries were detected)
- It uses a non-differentiable function (round/step function) which is basically approximated for the gradient calculation
- My findings should that it detects boundaries - but most of the time those boundaries could not easily be interpreted (like end/beginning of words etc.)
- I tried to create a metric based analysis - therefore I marked the expected boundaries in a text (e.g. start/end word etc.) and measured the differences of the detected boundaries - results were not very promising
- Maybe in a different settings (non-textual) the architecture would be more beneficial - or my implementation was just wrong ;)

### License
- None - if you really happen to use some of the code/documents/visualization - its nice if you link the repo ;)
