from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
from transformers import pipeline
import jsonlines
from tqdm import tqdm
import os
from openai import OpenAI

# from transformers import WhisperModel, WhisperFeatureExtractor


from fastapi import File, UploadFile
import time

from nnkh import translate

class TranscriptionRequest(BaseModel):
    # folder_path: str
    audio_file: UploadFile  # Change folder_path to audio_file
class TranscriptionResponse(BaseModel):
    file_path: str
    transcript: str



def save_uploaded_file(file: UploadFile):
    """Saves the uploaded file locally."""
    with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as buffer:
        buffer.write(file.file.read())


api_key = 'insert_private_key_here'
os.environ["OPENAI_API_KEY"] = 'insert_private_key_here'
client = OpenAI()
import openai
class PhoWhisperAgent:
    def __init__(self, model_name="vinai/PhoWhisper-tiny", device_id = 0):
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name, device = device_id)
        # self.transcriber.model.half()
        self.model_name = model_name

        # self.

    def inference(self, audio_file_path):
        # with open(audio_file_path, "rb") as audio_file:
        transcript = self.transcriber(audio_file_path)['text']
        # audio_file= open(audio_file_path, "rb")
        # transcript = client.audio.transcriptions.create(
        # model="whisper-1", 
        # file=audio_file
        # ).text
        # print(transcript)
        # # print(translation.text)
        return transcript
    
    def inference(self, audio_file_path):
        # with open(audio_file_path, "rb") as audio_file:
        # transcript = self.transcriber(audio_file_path)['text']
        audio_file= open(audio_file_path, "rb")
        transcript = client.audio.transcriptions.create(
        model="whisper-1", 
        file=audio_file
        ).text
        # print(transcript)
        # # print(translation.text)
        return transcript

    def batch_inference_with_path(self, audio_folder_path):
        audio_paths = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith('.wav')]
        transcriptions = []
        for audio_path in tqdm(audio_paths):
            transcription = self.inference(audio_path)
            file_name = os.path.basename(audio_path)
            transcriptions.append({'file_path': file_name, 'transcript': transcription})
        return transcriptions
    

UPLOAD_DIR = 'temp_storage'
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
ASR_AGENT = PhoWhisperAgent(model_name='vinai/PhoWhisper-tiny', device_id = 0)
GPT_INSTRUCTION =  'Đây là bản dịch một đoạn audio sang văn bản. Hãy sửa lại các từ sai chính tả và các từ không hợp ngữ cảnh. Không được bịa ra từ mới hoàn toàn, chỉ sửa lại các từ sai. Thêm dấu chấm và dấu phẩy cho phù hợp. Viết hoa chữ cái đầu và tên riêng'


app = FastAPI()


@app.post("/audio_transcribe/")
async def upload_and_transcribe_file(file: UploadFile = File(...)):
    """Endpoint to handle file upload and transcription.
    Input: Audio file (.wav, .mp3)
    
    Output: name of the file, the speech transcription and the time taken"""
    # agent = PhoWhisperAgent(model_name='vinai/PhoWhisper-small')

    start_time = time.time()  # Record start time
    save_uploaded_file(file)
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    transcription =    ASR_AGENT.inference(file_path)
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": GPT_INSTRUCTION},
            {"role": "user", "content": transcription}
        ]
    )
    corrected_sentence = completion.choices[0].message.content
    # corrected_sentence = transcription
    end_time = time.time()

    execution_time = end_time - start_time

    corrected_transcriptions = {'file_name': file.filename, 'transcript': corrected_sentence,'time':execution_time}

    return corrected_transcriptions


@app.post("/sign_lang_translate/")
async def sign_lang_translation(input_str: str):
    # agent = PhoWhisperAgent(model_name='vinai/PhoWhisper-small')

    """Endpoint to translate a normal vietnamese sentences into a list of sign language words.
    Input: input sentence
    
    Output: the sign language translation the time taken"""

    start_time = time.time()  # Record start time
    translation = translate(input_str)
    print(translation)
    end_time = time.time()

    execution_time = end_time - start_time

    corrected_transcriptions = {'translation': translation,'time':execution_time, 'original_sentence':input_str}

    return corrected_transcriptions



    


if __name__ == "__main__":



    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=3010, reload=True)
