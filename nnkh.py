import openai
from openai import OpenAI
import os
from tqdm import tqdm
openai.api_key = 'insert_key_here'
import pandas as pd
os.environ["OPENAI_API_KEY"] = 'insert_key_here'
import tiktoken
import pickle
from bpemb import BPEmb


from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.responses import JSONResponse
import jsonlines
from tqdm import tqdm
import os
from openai import OpenAI

import time
client = OpenAI()


class TranscriptionRequest(BaseModel):
    # folder_path: str
    input_sentence: str  # Change folder_path to audio_file
class TranscriptionResponse(BaseModel):
    output_sentence: str


with open('dict.pickle', 'rb') as f:
    sign_dict = pickle.load(f)

import numpy as np

from numpy import dot
from numpy.linalg import norm

words = []
sign_embeddings = []

for word in sign_dict.keys():
    words.append(word)
    sign_embeddings.append(sign_dict[word])
sign_embeddings = np.stack(sign_embeddings, axis = 0)
# print(sign_embeddings.shape)
# print(sign_dict.keys())

NUM_WORDS = sign_embeddings.shape[0]

def cosine_similarity(X, Y):
    cos_sim = (X * Y).sum(axis=1) / np.linalg.norm(X, axis=1) / np.linalg.norm(Y, axis=1)
    return cos_sim

bpemb_vi = BPEmb(lang="vi", vs=25000, dim = 300)


prompt_template = """Bạn có nhiệm vụ phiên dịch câu từ tiếng việt thông thường sang ngôn ngữ của người khiếm thính.
Bạn sẽ được cung cấp những quy luật biến đổi ngôn ngữ ký hiệu cơ bản ở phía dưới. 
Phân tích câu đầu vào, phân tích câu đó thuộc cấu trúc nào, các thành phần câu (chủ ngữ, vị ngữ, bổ ngữ, trạng ngữ, từ chỉ thời...) nằm ở đâu, và biến đổi theo quy luật

Kết quả trả ra không cần phân tích, chỉ cần ra câu kết quả dịch sang ngôn ngữ ký hiệu.
 Chỉ output ra câu trả lời, không có dấu mũi tên từ câu gốc. 
 Câu trả lời PHẢI có dấu phẩy ngăn cách giữa các từ ngôn ngữ ký hiệu
Không đưa ra câu trả lời tương tự "Xin lỗi, tôi không thể...". Nếu không thể dịch được, đơn giản hãy tách câu gốc ra thành các từ khác nhau.

** Quy luật biến đổi **
1. Những câu có cấu trúc S - V - O => được chuyển về thành  S - O - V (với S là chủ ngữ và đối tượng, V là động từ, O là bổ ngữ)  
Ví dụ: tôi ăn hai quả táo xanh --> Tôi, táo xanh, ăn, hai 


1.1. Câu có thành phần bổ ngữ và mở rộng bổ ngữ
Ở ngôn ngữ kí hiệu, khi mở rộng thành phần bổ ngữ, trật tự kí hiệu ở bổ ngữ sẽ thay đổi khác nhau tùy thuộc vào thành phần mà nó bổ túc. Nếu nó bổ túc cho chủ ngữ thì thành phần bổ túc sẽ đứng cạnh chủ ngữ. Nếu nó bổ túc cho vị ngữ thì sẽ đứng cạnh vị ngữ.
Ví dụ, câu có thành phần bổ ngữ bổ túc cho chủ ngữ
Ví dụ: tôi (chủ ngữ) đến (vị ngữ) siêu thị cùng mẹ (bổ ngữ có thảnh phần bổ túc cho chủ ngữ) --> tôi mẹ cùng (chủ ngữ) siêu thị (bổ ngữ) đến (vị ngữ)

Ví dụ, câu có thành phần bổ ngữ bổ túc cho vị ngữ
Tôi (Chủ ngữ) đến (vị ngữ) siêu thị bằng xe đạp (bổ ngữ có thành phần bổ túc cho vị ngữ) --> Tôi (chủ ngữ) siêu thị (bổ ngữ) đến (vị ngữ) xe đạp (thành phần bổ túc cho vị ngữ)

Ví dụ, khi câu chứa bổ ngữ có cả hai thành phần bổ túc trên sẽ có cấu trúc như sau
Ví dụ: Tôi đến siêu thị cùng mẹ bằng xe đạp --> Tôi mẹ cùng siêu thị đến xe đạp

1.2. Câu mở rộng thành phần vị ngữ (động từ)
Ví dụ: Tôi thích đến siêu thị cùng mẹ --> Tôi mẹ cùng siêu thị đến thích

1.3. Câu mở rộng thành phần chủ ngữ
Ví dụ: Bà nội của tôi (chủ ngữ) thích đi (vị ngữ) siêu thị cùng mẹ tôi (bổ ngữ) --> Bà nội của tôi (chủ ngữ) mẹ của tôi cùng siêu thị (bổ ngữ)  đi thích (vị ngữ(

1.4. Trật tự kí hiệu trong câu đơn phủ định 
Trong ngôn ngữ ký hiệu, từ phủ định (không, chẳng, chưa) đứng ở cuối câu.
Ví dụ: Tôi (chủ ngữ) không (phủ định) thích đến (vị ngữ) siêu thị (bổ ngữ) --> Tôi (chủ ngữ) siêu thị (bổ ngữ) đến thích (vị ngữ) không (phủ định)

1.5. Trật tự từ trong câu đơn có thành phần trạng ngữ 

1.5.a. Trạng ngữ chỉ thời gian: 
- Trong ngôn ngữ ký hiệu, trạng ngữ chỉ thời gian được chia thành hai kiểu như sau: nếu thời gian dài, có ý nghĩa tổng quát (buổi, ngày, tuần, tháng,…) thường sẽ đứng trước động từ
Ví dụ: Chủ nhật (trạng ngữ), tôi (chủ ngữ) đến (vị ngữ) siêu thị (bổ ngữ) --> Chủ nhật (trạng ngữ) tôi (chủ ngữ) siêu thị (bổ ngữ) đến (vị ngữ).

- Nếu thời gian ngắn, cụ thể hơn (giờ, phút) thường đứng sau động từ. Nếu xuất hiện hai phần của biểu đạt thời gian thì thường sẽ tách riêng như sau:
Ví dụ: Chủ nhật (trạng ngữ), tôi (chủ ngữ) đến siêu thị (vị ngữ & bổ ngữ) lúc 7 giờ tối (trạng ngữ) --> Chủ nhật tối (trạng ngữ) tôi (chủ ngữ) đến siêu thị (vị ngữ & bổ ngữ) 7 giờ (trạng ngữ).

- Cùng với trạng ngữ chỉ thời gian, yếu tố thì (thời) cũng có những kiểu biểu đạt rất đặc trưng với các kí hiệu “sẽ” chỉ thì tương lai và “đã” chỉ thì quá khứ đi. Các kí hiệu này thường có trật tự đứng cuối câu. 
Ví dụ: Chủ nhật tuần sau (trạng ngữ) tôi (chủ ngữ) sẽ đến (vị ngữ) siêu thị (bổ ngữ) --> tuần sau chủ nhật (trạng ngữ) tôi (chủ ngữ) siêu thị (bổ ngữ) đến (vị ngữ) sẽ (từ chỉ thời)

1.5.b. Trạng ngữ chỉ địa điểm: Ở ngôn ngữ kí hiệu, trạng ngữ chỉ địa điểm, nơi chốn thường xuất hiện ở đầu câu và ít có sự khác biệt với tiếng Việt.
Ví dụ: Ở trường (trạng ngữ), tôi (chủ ngữ) có bạn thân (vị ngữ) --> Ở trường (trạng ngữ), tôi (Chủ ngữ) bạn thân có (vị ngữ)

3. Các câu hỏi lựa chọn kiểu như “…đúng không/ phải không?” luôn đi kèm với sự biểu hiện trên nét mặt là cặp chân mày nhướng lên và đôi mắt hướng về phía người được hỏi biểu lộ sự chờ đợi một sự xác nhận
Ví dụ: Ngày mai là thứ ba đúng không? --> Mai thứ ba đúng (+ nét mặt)?

4. Câu hỏi có từ để hỏi như: ai, gì, mấy, thế nào, bao nhiêu, đâu, nào, tại sao…, =>  kí hiệu để hỏi luôn luôn đứng ở cuối câu
Ví dụ:
Ai cho bạn mượn sách? --> Sách, cho bạn, mượn, ai?
Em có bao nhiêu cái kẹo? --> Em, kẹo, có bao nhiêu?
Bạn thích ăn gì? --> Bạn, ăn, thích gì?
Gia đình của bạn có mấy người? --> Bạn, gia đình, người, mấy?


5.Trong cụm danh từ/ danh ngữ của ngôn ngữ kí hiệu, kí hiệu số lượng ứng với số từ trong ngôn ngữ nói tự nhiên bắt buộc phải đứng sau kí hiệu chỉ sự vật ứng với danh từ
Ví dụ:
Một con vịt --> Vịt một
Hai quả táo xanh --> Táo xanh hai

** Hết quy luật biến đổi **

Bạn sẽ được cho hai ví dụ sau

Câu đầu vào: hôm_qua, cô ấy đi học lúc 2 giờ tối

Phân tích: hôm qua (trạng ngữ) cô ấy (chủ ngữ) đi học (vị ngữ) lúc 2 giờ tối (trạng ngữ)
Câu này giống với quy tắc 1.5.a:  Nếu thời gian ngắn, cụ thể hơn (giờ, phút) thường đứng sau động từ. Nếu xuất hiện hai phần của biểu đạt thời gian thì thường sẽ tách riêng như sau:

Câu đầu ra: hôm_qua, tối, cô ấy, đi học, 2 giờ

Câu đầu vào: có ai ở đây không?
Phân tích : câu này là câu có từ để hỏi. 4.. Câu hỏi có từ để hỏi như: ai, gì, mấy, thế nào, bao nhiêu, đâu, nào, tại sao…, =>  kí hiệu để hỏi luôn luôn đứng ở cuối câu

Câu đầu ra: ở đây, ai?

Câu đầu vào: việt_nam có lũ_lụt không?
Phân tích : câu này là câu Các câu hỏi lựa chọn kiểu như “…đúng không/ phải không?” luôn đi kèm với sự biểu hiện trên nét mặt là cặp chân mày nhướng lên và đôi mắt hướng về phía người được hỏi biểu lộ sự chờ đợi một sự xác nhận
Câu đầu ra: việt_nam, lũ_lụt, (+nét mặt)


Câu đầu vào: chú của tôi 50 tuổi, thấp
Phân tích : câu này là câu có cấu trúc S - V - O => được chuyển về thành  S - O - V (với S là chủ ngữ và đối tượng, V là động từ, O là bổ ngữ)  
Câu đầu ra:  Chú, của tôi, tuổi, 50, thấp 

Câu đầu vào: Tôi không có xe_máy nhưng có xe_đạp.
Câu đầu ra:  Tôi,xe_máy, không, xe_đạp, có. 

Câu đầu vào: 1 năm có 4 mùa: Xuân, Hạ, Thu, Đông.
Câu đầu ra:  1, năm, mùa , 4, Xuân, Hạ, Thu, Đông. 

Câu đầu vào: Tôi không thích sầu_riêng vì nó hôi
Câu đầu ra:  Tôi, sầu_riêng, không thích , vì, nó, hôi, (+ nét mặt)
Đó là 3 ví dụ. 

Kết quả trả ra không cần phân tích, chỉ cần ra câu kết quả dịch sang ngôn ngữ ký hiệu. Chỉ output ra câu trả lời, không có dấu mũi tên từ câu gốc. 
 Câu trả lời PHẢI có dấu phẩy ngăn cách giữa các từ ngôn ngữ ký hiệu
Không đưa ra câu trả lời tương tự "Xin lỗi, tôi không thể...". Nếu không thể dịch được, đơn giản hãy tách câu gốc ra thành các từ khác nhau.


Câu đầu vào:   """


def truncate_text_tokens(text, max_tokens=20000):
    encoding = tiktoken.encoding_for_model("gpt-4-1106-preview")
    text_encoded = encoding.encode(text)
    text_encoded_truncated  =text_encoded[:max_tokens]
    truncated_text = encoding.decode(text_encoded_truncated)


    # print(len(text_encoded), len(text_encoded_truncated))

    return truncated_text


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
 
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


import underthesea
from underthesea import word_tokenize




def translate(text):
    text = text.lower()
    truncated_text = truncate_text_tokens(text)
    # print(truncated_text)
    truncated_text = word_tokenize(truncated_text, format="text")
    # truncated_text 
    # print(word_tokenize('có ai ở đây không?', format="text"))
    # print(word_tokenize('hôm qua, cô ấy đi học lúc 2 giờ tối', format="text"))
    # print(word_tokenize('việt nam có lũ lụt không', format="text"))

    truncated_text = truncated_text.replace('_', ' ')

    print(truncated_text)

    # print(truncated_text)
    # print(truncated_text)
    # try:
    response =completion_with_backoff(
    model="gpt-4-1106-preview",
    messages=[
        {
        "role": "system",
        "content": prompt_template
        },
        {
        "role": "user",
        "content": truncated_text
        },
    ],
    temperature= 0.2,
    max_tokens=100,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=0
    )

    answer = response.choices[0].message.content

    split_answer = answer.split(',')
    final_answer = []
    for word in split_answer:
        print(word)
        print(final_answer)
        if 'nét mặt' not in word:
            if word in ['tôi','mình', 'bọn mình', 'chúng mình', 'chúng tôi'] :
                synonym = 'chúng tôi'
                final_answer.append(synonym)

            elif word in ['bạn','cậu']:
                synonym = 'bạn'
                final_answer.append(synonym)

            elif word in ['các bạn','mọi người','các cậu']:
                synonym = 'các bạn'
                final_answer.append(synonym)

            else:
                ids  = bpemb_vi.encode_ids(word)
                embedding = bpemb_vi.vectors[ids]
                embedding = np.mean(embedding, axis=0)
                embedding = np.expand_dims(embedding,axis = 0)
                embedding = np.repeat(embedding,NUM_WORDS,axis = 0)
                # print(embedding.shape)
                cos_sim = cosine_similarity(embedding, sign_embeddings)
                # print(cos_sim)
                synonym_index = cos_sim.argsort()[-1:][::-1][0]
                synonym = words[synonym_index]
                score = cos_sim[synonym_index]
                final_answer.append(synonym)

        else:
            if 'có' in answer:
                synonym = 'có ... không'
                final_answer.append(synonym)

                final_answer.remove('có')
            elif 'đúng' in answer:
                synonym = 'đúng ... không'
                final_answer.append(synonym)

                final_answer.remove('đúng')

            elif 'phải' in answer:
                synonym = 'phải ... không'
                final_answer.append(synonym)

                final_answer.remove('phải')

            elif 'nên' in answer:
                synonym = 'nên ... không'
                final_answer.append(synonym)

                final_answer.remove('nên')

            elif 'cần' in answer:
                synonym = 'cần ... không'
                final_answer.append(synonym)

                final_answer.remove('cần')

            elif 'muốn' in answer:
                synonym = 'muốn ... không'
                final_answer.append(synonym)
                final_answer.remove('muốn')

            elif 'sợ' in answer:
                synonym = 'sợ ... không'
                final_answer.append(synonym)

                final_answer.remove('sợ')

            elif 'mệt' in answer:
                synonym = 'mệt ... không'
                final_answer.append(synonym)

                final_answer.remove('mệt')

            elif 'đói' in answer:
                synonym = 'đói ... không'
                final_answer.append(synonym)

                final_answer.remove('đói')

            elif 'sợ' in answer:
                synonym = 'sợ ... không'
                final_answer.append(synonym)

                final_answer.remove('sợ')

            else:
                synonym = 'có ... không'
                final_answer.append(synonym)

                final_answer.remove('có')


    # answer = ', '.join(final_answer)
        # print(cos_sim.shape)
        # print(word, synonym, score)
        # print(cos_sim)
        # print(word,embedding.shape)

        # print(word)
    return answer
    # except Exception as e:
    #     return f"Error: {e}"

# Example use with a placeholder text
import json

# # Load the JSON data
# with open('nuoi_tom.json', 'r', encoding='utf-8') as file:
#     data = json.load(file)

# # Iterate over each item and process it
# for item in data:
#     full_text = item['full_text']  # Assuming this is how your data is structured
#     extracted_info = extract_technology_information(full_text)
#     print(extracted_info)


import tqdm
import numpy as np

# app = FastAPI()


# @app.post("/translate/")
# async def sign_lang_translation(input_str: str):
#     """Endpoint to handle file upload and transcription.
#     Input: Audio file (.wav, .mp3)
    
#     Output: name of the file, the speech transcription and the time taken"""
#     # agent = PhoWhisperAgent(model_name='vinai/PhoWhisper-small')

#     start_time = time.time()  # Record start time
#     translation = translate(input_str)
#     print(translation)
#     end_time = time.time()

#     execution_time = end_time - start_time

#     corrected_transcriptions = {'translation': translation,'time':execution_time, 'original_sentence':input_str}

#     return corrected_transcriptions



# if __name__ == "__main__":



#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=3010, reload=True)
