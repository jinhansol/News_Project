import os
import requests
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ✅ 환경 변수 로딩
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ LLM 설정 (LangChain)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)

# ✅ 개별 기사 요약용 프롬프트
summary_prompt = PromptTemplate.from_template("""
너는 한국어 뉴스 기사를 요약하는 AI야.
아래 기사 내용을 사용자가 원하는 목적에 맞춰 3~4줄로 요약해줘.

요약 목적: {purpose}

뉴스 기사 원문:
{article}
""")
summary_chain = summary_prompt | llm

# ✅ 종합 트렌드 요약용 프롬프트
trend_prompt = PromptTemplate.from_template("""
다음은 최근 뉴스 기사들의 요약입니다.
요약들을 바탕으로 '{purpose}'에 대한 최신 트렌드와 공통 내용을 정리해서 사용자 입장에서 보기 쉽게,
3~5줄 분량으로 종합 요약을 해줘.

기사 요약 모음:
{summaries}
""")
trend_chain = trend_prompt | llm

# ✅ FastAPI 앱 초기화
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ✅ 키워드 → 목적 매핑
KEYWORD_PURPOSE_MAP = {
    "코인": "가상자산 뉴스 요약",
    "비트코인": "가상자산 뉴스 요약",
    "가상화폐": "가상자산 뉴스 요약",
    "이더리움": "가상자산 뉴스 요약",

    "취업": "취업 시장 동향 요약",
    "취준": "취업 준비 트렌드 요약",
    "스펙": "청년 구직 전략 정보 요약",
    "공채": "대기업 채용 동향 요약",
    "채용": "채용 뉴스 요약",
    "NCS": "공기업 준비 뉴스 요약",

    "주식": "주식시장 뉴스 요약",
    "ETF": "금융상품 동향 요약",
    "미국 증시": "해외 증시 동향 요약",
    "테슬라": "해외 기업 뉴스 요약",
    "애플": "글로벌 IT 기업 뉴스 요약",

    "부동산": "부동산 시장 동향 요약",
    "월세": "주거비용/부동산 트렌드 요약",
    "아파트": "주택 시장 뉴스 요약",

    "다이어트": "헬스/운동 트렌드 요약",
    "헬스": "헬스케어 트렌드 요약",
    "운동": "MZ 라이프스타일 뉴스 요약",

    "메타버스": "신기술 트렌드 요약",
    "일론 머스크": "글로벌 기업가 뉴스 요약",
    "챗GPT": "AI 기술 트렌드 요약",
    "AI": "인공지능 뉴스 요약"
}


def map_user_keyword_to_purpose(keyword: str) -> str:
    for k, purpose in KEYWORD_PURPOSE_MAP.items():
        if k in keyword:
            return purpose
    return f"{keyword} 관련 뉴스 요약"  # fallback용


# ✅ 기사 링크 크롤링
def crawl_news_links_by_keyword(keyword: str, max_articles=5):
    search_url = f"https://search.naver.com/search.naver?where=news&query={keyword}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    links = []

    # 1. a.news_tit → a.tit → 전체 <a> 중 naver 주소
    for a_tag in soup.select('a.news_tit'):
        title = a_tag.get("title", "").strip() or a_tag.get_text(strip=True)
        href = a_tag.get("href", "").strip()
        if href.startswith("http"):
            links.append({"title": title, "url": href})
        if len(links) >= max_articles:
            return links

    for a_tag in soup.select('a.tit'):
        title = a_tag.get("title", "").strip() or a_tag.get_text(strip=True)
        href = a_tag.get("href", "").strip()
        if href.startswith("http"):
            links.append({"title": title, "url": href})
        if len(links) >= max_articles:
            return links

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        title = a_tag.get("title", "").strip() or a_tag.get_text(strip=True)
        if "news.naver.com" in href:
            links.append({"title": title, "url": href})
        if len(links) >= max_articles:
            return links

    if not links:
        raise HTTPException(status_code=404, detail="관련 뉴스 기사를 찾을 수 없습니다.")
    return links


# ✅ 본문 추출
def extract_article_text(url: str):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.content, "html.parser")

    selectors = [
        ("div", {"id": "dic_area"}),  # 네이버
        ("div", {"class": "article_body"}),  # 중앙일보
        ("div", {"class": "article-text"}),  # 한겨레
        ("div", {"class": "story-news article"})  # 연합뉴스
    ]

    for tag, attrs in selectors:
        node = soup.find(tag, attrs=attrs)
        if node:
            return node.get_text(strip=True)

    paragraphs = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
    if not text:
        raise HTTPException(status_code=400, detail="본문 추출 실패")
    return text


# ✅ 요약 실행 함수
def summarize_with_langchain(article_text: str, purpose: str):
    result = summary_chain.invoke({"purpose": purpose, "article": article_text})
    return result.content


# ✅ 트렌드 종합 요약 실행 함수
def summarize_trend_digest(article_summaries: list, purpose: str):
    joined = "\n".join(article_summaries)
    result = trend_chain.invoke({"purpose": purpose, "summaries": joined})
    return result.content


# ✅ FastAPI API 엔드포인트
@app.post("/news_trend/")
async def news_trend(request: Request):
    data = await request.json()
    keyword = data.get("keyword", "")
    if not keyword:
        raise HTTPException(status_code=422, detail="keyword는 필수입니다.")

    # 1) 검색 목적 추출
    purpose = map_user_keyword_to_purpose(keyword)

    # 2) 기사 링크 크롤링
    links = crawl_news_links_by_keyword(keyword, max_articles=5)

    # 3) 기사별 본문 → 요약
    results = []
    all_summaries = []

    for news in links:
        try:
            article = extract_article_text(news["url"])
            summary = summarize_with_langchain(article, purpose)
        except Exception as e:
            summary = f"요약 실패: {e}"
        all_summaries.append(summary)
        results.append({
            "title": news["title"],
            "url": news["url"],
            "summary": summary
        })

    # 4) 종합 요약 생성
    trend_summary = summarize_trend_digest(all_summaries, purpose)

    # 5) 응답
    return {
        "keyword": keyword,
        "purpose": purpose,
        "trend_digest": trend_summary,  # ✅ 종합 요약 출력
        "trend_articles": results       # ✅ 개별 기사 리스트
    }
