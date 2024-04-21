# from fastapi import FastAPI, File, UploadFile
# from pydantic import BaseModel
# from typing import List
# from fastapi.responses import JSONResponse
from transformers import pipeline
import jsonlines
from tqdm import tqdm
import os
from openai import OpenAI

# from transformers import WhisperModel, WhisperFeatureExtractor


# from fastapi import File, UploadFile
import time

from nnkh import translate



api_key = "insert_key_here"
os.environ["OPENAI_API_KEY"] =  "insert_key_here"
client = OpenAI()


GPT_INSTRUCTION =  'Đây là bản dịch một đoạn audio sang văn bản. Hãy sửa lại các từ sai chính tả và các từ không hợp ngữ cảnh. Không được bịa ra từ mới hoàn toàn, chỉ sửa lại các từ sai. Thêm dấu chấm và dấu phẩy cho phù hợp. Viết hoa chữ cái đầu và tên riêng'

import openai
class PhoWhisperAgent:
    def __init__(self, model_name="vinai/PhoWhisper-tiny", device_id = 0):
        self.transcriber = pipeline("automatic-speech-recognition", model=model_name, device = device_id)
        # self.transcriber.model.half()
        self.model_name = model_name

        # self.

    def inference(self, audio_file_path, use_gpt_correction  =True):
        # with open(audio_file_path, "rb") as audio_file:
        transcript = self.transcriber(audio_file_path)['text']
        if use_gpt_correction:
                completion = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": GPT_INSTRUCTION},
                        {"role": "user", "content": transcription}
                    ]
                )
                corrected_sentence = completion.choices[0].message.content
                transcript = corrected_sentence

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
        if use_gpt_correction:
                completion = client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "system", "content": GPT_INSTRUCTION},
                        {"role": "user", "content": transcription}
                    ]
                )
                corrected_sentence = completion.choices[0].message.content
                transcript = corrected_sentence

        return transcript
    

    def batch_inference_with_path(self, audio_folder_path):
        audio_paths = [os.path.join(audio_folder_path, file) for file in os.listdir(audio_folder_path) if file.endswith('.wav')]
        transcriptions = []
        for audio_path in tqdm(audio_paths):
            transcription = self.inference(audio_path)
            file_name = os.path.basename(audio_path)
            transcriptions.append({'file_path': file_name, 'transcript': transcription})
        return transcriptions
    



