from detoxify import Detoxify

def check_comment_toxicity(comment):
    model = Detoxify('original')

    results = model.predict(comment)

    if results['toxicity'] > 0.5:
        return "The comment is toxic."
    else:
        return "The comment is not toxic."


user_comment = input("Enter a comment: ")

result = check_comment_toxicity(user_comment)
print(result)
