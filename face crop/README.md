
# face_crop

### 1. Usage
環境に合わせてPATHの設定をして下さい。実行するとimage_pathで設定したフォルダー内の画像全てがcropされ、out_jpgに保存されます。
+ image_path (cropしたい画像が入ったフォルダー)
+ out_jpg (保存先)
+ cascade_path (haarcascade_frontalface_alt.xmlの場所)

### 2. Demo
![Demo](https://github.com/baibai25/deepface/blob/master/face%20crop/Peek%202017-09-22%2014-50.gif)

画像に複数人写っている場合、正しくcropされない場合があります。その時は、detectMultiScaleの部分のパラメータを調整して下さい。http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html


# web_camera.py

### 1. Usage
こちらも同様に保存先とhaarcascade_frontalface_alt.xmlの場所を設定して下さい。ipynb形式でも実行ができますが、挙動が不安定なのでおすすめしません。
実行は以下の通りです。

+ ターミナルを開き、python web_camera.pyと入力し実行します。
+ カメラが起動し、カメラの映像が表示されます。
+ キーボードの、's'で写っている顔が画像で保存されます。'q'でカメラを停止します。
