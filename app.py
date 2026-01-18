from ultralytics import YOLO
import gradio as gr

model = YOLO('D:/Проекты/Python/jupyter_notebooks/detection_faces/best_model_phash.pt')

def detect_face_gender(image):
    results = model(image)
    return results[0].plot()

iface = gr.Interface(
    fn=detect_face_gender,
    inputs=gr.Image(type='pil'),
    outputs=gr.Image(type='pil'),
    title='Face Detection',
    description='Загрузите изображение - модель найдет лица и определит пол (man/woman).'
)

if __name__ == '__main__':
    iface.launch(share=True)