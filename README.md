# BLSTM-CRF-NER
## reference:

paper:

Neural Architectures for Named Entity Recognition<br/>
End-toEnd Sequence labeling via BLSTM-CNN-CRF<br/>
   
code:

https://github.com/ZhixiuYe/NER-pytorch<br/>

## requirement

python3.6

pytorch

## usage:

train model:

```
python train.py
```

query:

```
python debug.py
```

## File orgnization

```
|- train.py 
|- debug.py 
|- [dir] dataset (word library)
|- [dir] evaluation (help tools when training)
|- [dir] models (well-trained models)
```