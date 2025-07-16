import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import openai

# 환경변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=api_key)

# 테스트용 뉴스 페이지 크롤링
url = "https://news.ycombinator.com/"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
title = soup.select_one("title").get_text(strip=True)
print("사이트 제목:", title)

# OpenAI 요약 테스트
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "기사 내용을 요약해줘."}
    ]
)
print("AI 요약 결과:", response.choices[0].message.content)
