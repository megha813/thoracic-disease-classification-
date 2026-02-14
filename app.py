import gradio as gr
import requests
import pandas as pd
import io

# FastAPI base URL
BASE_URL = "http://127.0.0.1:8000"

# ⚠️ Make sure this EXACTLY matches your FastAPI route
PREDICT_URL = f"{BASE_URL}/predict_from_png/"
GET_JSON_URL = f"{BASE_URL}/json/"

def predict_xray(image, threshold):
    """
    Sends PNG image to FastAPI /predict_from_png/
    Then fetches predictions from /json/{file_id}
    Applies threshold dynamically.
    """

    if image is None:
        return pd.DataFrame(
            [["No image uploaded", 0, "Error"]],
            columns=["Disease", "Probability", "Prediction"]
        )

    try:
        # Convert PIL image to bytes (no temp file needed)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Step 1: Send image to FastAPI
        response = requests.post(
            PREDICT_URL,
            files={"file": ("image.png", img_bytes, "image/png")},
            timeout=180
        )

        # if response.status_code != 200:
        #     return pd.DataFrame(
        #         [[f"API Error {response.status_code}", 0, "Error"]],
        #         columns=["Disease", "Probability", "Prediction"]
        #     )

        # result = response.json()
        # file_id = result.get("file_id")

        # if not file_id:
        #     return pd.DataFrame(
        #         [["No file_id returned", 0, "Error"]],
        #         columns=["Disease", "Probability", "Prediction"]
        #     )

        # # Step 2: Fetch predictions JSON
        # json_response = requests.get(
        #     f"{GET_JSON_URL}{file_id}",
        #     timeout=60
        # )

        # if json_response.status_code != 200:
        #     return pd.DataFrame(
        #         [["Error fetching JSON", 0, "Error"]],
        #         columns=["Disease", "Probability", "Prediction"]
        #     )

        probabilities = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(
            list(probabilities.items()),
            columns=["Disease", "Probability"]
        )

        df["Probability"] = df["Probability"].astype(float)

        # Apply threshold
        df["Prediction"] = df["Probability"].apply(
            lambda x: "Positive" if x >= threshold else "Negative"
        )

        # Sort highest first
        df = df.sort_values(by="Probability", ascending=False)

        return df

    except Exception as e:
        return pd.DataFrame(
            [[str(e), 0, "Error"]],
            columns=["Disease", "Probability", "Prediction"]
        )


with gr.Blocks() as demo:

    gr.Markdown("# 🩺 Chest X-Ray Disease Prediction (MedViT)")

    with gr.Row():

        # LEFT SIDE → Image
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray (PNG)",
                height=450
            )

        # RIGHT SIDE → Controls + Output
        with gr.Column(scale=1):

            threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.5,
                step=0.01,
                label="Threshold"
            )

            predict_btn = gr.Button("Predict")

            output_table = gr.Dataframe(
                headers=["Disease", "Probability", "Prediction"],
                datatype=["str", "number", "str"],
                interactive=False,
                height=350
            )

    predict_btn.click(
        fn=predict_xray,
        inputs=[image_input, threshold_slider],
        outputs=output_table
    )

demo.launch(share= True)
