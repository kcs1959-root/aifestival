# insta-stars
だいたいこれと同じ  
https://github.com/munky69rock/mnist-demo  
http://qiita.com/munky69rock/items/8eef640723ce4f948e61


## 遊び方
### 1. 学習する

```
$ python train.py
```

とすると、`cache/` にチェックポイントファイルがたくさんできます

### 2. Web サーバーを起動

canvas にお絵かきして文字の判定をしたり、カメラで画像をとって判定したいときは Web サーバーを起動しましょう。

```
$ FLASK_APP=index.py flask run
```

ブラウザで `http://127.0.0.1:5000/` にアクセスして遊びましょう。Chrome だとうまく動かない :innocent: ので、firefox などを使いましょう。

### 3. Twitter で遊ぶ
Twitter Developers で Consumer Token や Access Token などを発行したら以下の内容で `credentials.py` を作りましょう。

```
CONSUMER_TOKEN = "*******"
CONSUMER_SECRET = "*************"
ACCESS_TOKEN = "*********"
ACCESS_TOKEN_SECRET = "***********"
```

```
$ python twitter.py
```

として、自分のTwitterアカウントに画像つきリプライを飛ばすとターミナルに判定結果が表示されます。
