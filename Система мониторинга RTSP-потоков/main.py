import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  

from modules.initialization import Initialization
from modules.video_processor import VideoProcessor
from modules.gradio_interface import GradioInterface

def main():
    # Инициализация
    init = Initialization()

    # VideoProcessor с передачей пути к базе данных
    video_processor = VideoProcessor(init.DB_PATH)

    # Gradio интерфейса
    gradio_interface = GradioInterface(video_processor)

    # Gradio
    demo = gradio_interface.create_interface()
    demo.launch(share=True)  

if __name__ == "__main__":
    main()