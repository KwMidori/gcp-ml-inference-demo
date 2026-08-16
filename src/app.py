def predict_priority(text: str) -> str:
    text = text.lower()

    urgent_words = ["urgent", "immediately", "asap", "至急", "緊急"]

    for word in urgent_words:
        if word in text:
            return "high"

    return "normal"


if __name__ == "__main__":
    text = input("問い合わせ内容を入力してください: ")
    priority = predict_priority(text)
    print(f"優先度: {priority}")