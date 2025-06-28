import gradio as gr
from src.model.model import SummarizerModel
import torch

MODELS = {
    "CNN/DailyMail": "cnn_dailymail_epoch_5.pth",
    "XSum": "xsum_epoch_5.pth"
}

class SummarizerApp:
    def __init__(self):
        self.models = {}
        for name, path in MODELS.items():
            model = SummarizerModel()
            model.load_state_dict(torch.load(f"models/{path}"))
            self.models[name] = model
        
        self.current_model = self.models["CNN/DailyMail"]

    def summarize(self, text, model_name):
        self.current_model = self.models[model_name]
        return self.current_model.generate_summary(text)

app = SummarizerApp()

# Example texts
EXAMPLES = [
    [
        """ LONDON, England (Reuters) -- Harry Potter star Daniel Radcliffe gains access to a reported £20 million ($41.1 million) fortune as he turns 18 on Monday, but he insists the money won't cast a spell on him. Daniel Radcliffe as Harry Potter in "Harry Potter and the Order of the Phoenix" To the disappointment of gossip columnists around the world, the young actor says he has no plans to fritter his cash away on fast cars, drink and celebrity parties. "I don't plan to be one of those people who, as soon as they turn 18, suddenly buy themselves a massive sports car collection or something similar," he told an Australian interviewer earlier this month. "I don't think I'll be particularly extravagant. "The things I like buying are things that cost about 10 pounds -- books and CDs and DVDs." At 18, Radcliffe will be able to gamble in a casino, buy a drink in a pub or see the horror film "Hostel: Part II," currently six places below his number one movie on the UK box office chart. Details of how he'll mark his landmark birthday are under wraps. His agent and publicist had no comment on his plans. "I'll definitely have some sort of party," he said in an interview. "Hopefully none of you will be reading about it." Radcliffe's earnings from the first five Potter films have been held in a trust fund which he has not been able to touch. Despite his growing fame and riches, the actor says he is keeping his feet firmly on the ground. "People are always looking to say 'kid star goes off the rails,'" he told reporters last month. "But I try very hard not to go that way because it would be too easy for them." His latest outing as the boy wizard in "Harry Potter and the Order of the Phoenix" is breaking records on both sides of the Atlantic and he will reprise the role in the last two films. Watch I-Reporter give her review of Potter's latest » . There is life beyond Potter, however. The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. He will also appear in "December Boys," an Australian film about four boys who escape an orphanage. Earlier this year, he made his stage debut playing a tortured teenager in Peter Shaffer's "Equus." Meanwhile, he is braced for even closer media scrutiny now that he's legally an adult: "I just think I'm going to be more sort of fair game," he told Reuters. E-mail to a friend . Copyright 2007 Reuters. All rights reserved.This material may not be published, broadcast, rewritten, or redistributed.""",
        "CNN/DailyMail"
    ]
]

with gr.Blocks(css="footer {display: none}") as demo:
    gr.Markdown(
        """
        # 📝 Text Summarizer
        This app uses T5-small model fine-tuned on CNN/DailyMail and XSum datasets to generate summaries.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=4):
            text = gr.Textbox(
                label="Text to summarize",
                placeholder="Enter your text here...",
                lines=8
            )
            
        with gr.Column(scale=1):
            model_choice = gr.Radio(
                choices=list(MODELS.keys()),
                value="CNN/DailyMail",
                label="Model"
            )
            
    summary = gr.Textbox(label="Summary", lines=3)
    
    with gr.Row():
        clear = gr.Button("Clear")
        button = gr.Button("Summarize", variant="primary")
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=[text, model_choice],
        outputs=summary,
        fn=app.summarize,
        cache_examples=True
    )
    
    button.click(app.summarize, inputs=[text, model_choice], outputs=summary)
    clear.click(lambda: ["", "CNN/DailyMail"], outputs=[text, model_choice])

if __name__ == "__main__":
    demo.launch()


