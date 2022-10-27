# DNN_scratch
アーキテクチャでtensorflowやpytorchなどの深層学習のライブラリを使わず、DNNの実装を行いました。
reluは正の値しか残しませんが、負の値も残すmreluの実装も行っています。具体的な実装として、正の値だけフィルタしたものと負の値だけフィルタしたものをくっつけています。(サイズは二倍になります)

## Prerequires
- tensorflow(mnistのデータの取得に利用)
- matplotlib
- numpy

## Dataset
- mnist
- circle(与えられたデータが円の中に存在するかどうかを区別する)

## Training
```
# mnistを relu で訓練 
python main.py --mnist --relu

# circleを mrelu で訓練 
python main.py --circle --mrelu
```
