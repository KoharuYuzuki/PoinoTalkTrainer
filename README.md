# PoinoTalk Trainer
PoinoTalk Engineのモデルトレーナー

## 使い方

### 1. 環境構築
次のソフトウェアをインストールします  
- rye
- Python 3.10.x
- OpenJTalk (Pythonで音声合成をする場合のみ)

#### 補足
- Pythonはryeを使用してインストールしてください

### 2. 依存パッケージのインストール
次のコマンドで依存パッケージをインストールします  
```
$ rye sync
```

### 3. ファイルの配置
labディレクトリにラベルファイルを配置します  
wavディレクトリにWAVファイルを配置します  

#### 補足
- ラベルファイルとWAVファイルは拡張子以外のファイル名は同一にしてください
- ラベルファイルはOpenJTalkスタイルのものを使用してください
- WAVファイルはコンフィグファイルの"wav_fs"と同一のサンプリングレートのものを使用してください
- WAVファイルは任意のコーデックのものを使用できます
- ラベルファイルの作成にはSegKit-Dockerが便利です
- SegKit-Docker: https://github.com/KoharuYuzuki/SegKit-Docker

### 4. コンフィグファイルの作成
テンプレートファイル (config_template.jsonc) を参考に、コンフィグファイル (config.json) を作成します  
コンフィグの詳細はテンプレートファイルにコメントしてあります  

#### 補足
- テンプレートファイルは拡張子が.jsonc (コメント付きJSON) ですが、コンフィグファイルの拡張子は.jsonを使用してください

### 5. トレーニング
次のコマンドでモデルをトレーニングします  
```
$ python ./src/poinotalktrainer/train.py
```

### 6. 音声合成
次のコマンドで音声を合成します  
```
$ python ./src/poinotalktrainer/synth.py "読み上げるテキスト" "出力ファイル名.wav"
```

## Q&A

### Q. トレーニング済みモデルのライセンスは？
A. PoinoTalk Trainerを使用してトレーニングしたモデルであっても、あなたがモデルの制作者である場合は任意のライセンスを適用できます  
ただし、トレーニングに使用したラベルファイルやWAVファイルにライセンス等が設定されている場合にはそれに従ってください  
また、他人が配布しているモデルなどについては、権利者等に確認してください  

## ライセンス
PoinoTalk Licence 1.0  
https://raw.githubusercontent.com/KoharuYuzuki/PoinoTalkLicence/main/1.0.txt  
