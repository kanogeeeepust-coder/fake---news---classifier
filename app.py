import joblib
import gradio as gr

# Load trained model and vectorizer
model = joblib.load("/content/best_model.pkl")
vectorizer = joblib.load("/content/tfidf_vectorizer.pkl")


# Prediction function
def predict_news(text):

    # Vectorize input text
    text_vec = vectorizer.transform([text])

    # Predict label
    prediction = model.predict(text_vec)[0]

    # Predict probabilities
    prob = model.predict_proba(text_vec)[0]

    confidence = {
        "Fake": float(prob[0]),
        "Real": float(prob[1])
    }

    return prediction, confidence


# Gradio Interface
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=8,
        placeholder="Paste a news article here..."
    ),
    outputs=[
        gr.Text(label="Prediction"),
        gr.Label(label="Confidence Score")
    ],
    title="Fake News Detection System",
    description="This application predicts whether a news article is Fake or Real using a trained Logistic Regression model.",
    
    examples=[
        ["Breaking: Government announces new economic policy to boost employment."],
        ["Shocking! Scientists confirm aliens built pyramids in Egypt."],
        ["The central bank increased interest rates to control inflation."]
    ]
)

# Launch app
demo.launch()