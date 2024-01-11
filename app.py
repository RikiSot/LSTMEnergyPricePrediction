import gradio as gr
from predictions import predict


# Create title, description and article strings
title = "Iberian energy prices predictor 💸💸"
description = "A multivariate LSTM model that takes 72h of data and predicts the next 24h of prices"
article = "..."

# Create the Gradio demo
demo = gr.Interface(fn=predict,  # mapping function from input to output
                    inputs=gr.Textbox(lines=1, placeholder='Enter date in YYYY-MM-DD format', label='Enter date'),
                    outputs=[gr.Plot(),  # plot for predictions
                             gr.Number(label="Prediction time (s)")],  # prediction time
                    title=title,
                    description=description,
                    article=article)

# Launch the demo!
demo.launch(debug=True,
            share=True)
