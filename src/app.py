import joblib


model = joblib.load("model.joblib")

text = input("問い合わせ内容を入力してください: ")

prediction = model.predict([text])

print(f"予測結果: {prediction[0]}")