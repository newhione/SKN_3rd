"""
ChromaDB 벡터 데이터베이스 구축
전처리된 JSON 데이터를 임베딩하여 벡터 DB에 저장
"""

import json
import os
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)


def load_preprocessed_data(filepath):
    """
    전처리된 JSON 데이터 로드
    
    Args:
        filepath: JSON 파일 경로
        
    Returns:
        list: 정책 데이터 리스트
    """
    print(f"📂 데이터 로드 중: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"✅ 총 {len(data)}개의 정책 데이터 로드 완료")
    return data


def create_policy_text(policy):
    """
    정책 데이터를 임베딩을 위한 텍스트로 변환
    
    Args:
        policy: 정책 딕셔너리
        
    Returns:
        str: 결합된 텍스트
    """
    # 주요 필드들을 결합하여 검색 가능한 텍스트 생성
    text_parts = []
    
    if policy.get('정책명'):
        text_parts.append(f"정책명: {policy['정책명']}")
    
    if policy.get('정책설명'):
        text_parts.append(f"정책설명: {policy['정책설명']}")
    
    if policy.get('지원내용'):
        text_parts.append(f"지원내용: {policy['지원내용']}")
    
    if policy.get('대분류'):
        text_parts.append(f"대분류: {policy['대분류']}")
    
    if policy.get('중분류'):
        text_parts.append(f"중분류: {policy['중분류']}")
    
    if policy.get('정책키워드'):
        text_parts.append(f"키워드: {policy['정책키워드']}")
    
    # 자격 조건 (검색 정확도 향상)
    if policy.get('추가자격조건'):
        # 너무 길면 앞부분만
        qual = policy['추가자격조건'][:300]
        text_parts.append(f"자격조건: {qual}")
    
    # 연령 제한
    min_age = policy.get('지원최소연령', '0')
    max_age = policy.get('지원최대연령', '0')
    if min_age != '0' or max_age != '0':
        age_info = f"연령: {min_age}세 ~ {max_age}세"
        text_parts.append(age_info)
    
    # 지원금액
    min_amount = policy.get('최소지원금액', '0')
    max_amount = policy.get('최대지원금액', '0')
    if min_amount != '0' or max_amount != '0':
        amount_info = f"지원금액: {min_amount}원 ~ {max_amount}원"
        text_parts.append(amount_info)
    
    # 텍스트가 비어있으면 최소한의 정보라도 포함
    if not text_parts:
        text_parts.append(f"정책 데이터")
    
    return "\n".join(text_parts)


def get_embedding(text, model="text-embedding-3-small"):
    """
    OpenAI API를 사용하여 텍스트 임베딩 생성
    
    Args:
        text: 임베딩할 텍스트
        model: 사용할 임베딩 모델
        
    Returns:
        list: 임베딩 벡터
    """
    # 텍스트 정제
    text = text.replace("\n", " ").strip()
    
    # 빈 텍스트 체크
    if not text or len(text) < 3:
        text = "정책 정보"
    
    # 너무 긴 텍스트는 잘라내기 (토큰 제한)
    if len(text) > 8000:
        text = text[:8000]
    
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def build_chromadb(policies, db_path="../data/vectordb"):
    """
    ChromaDB 벡터 데이터베이스 구축
    
    Args:
        policies: 정책 데이터 리스트
        db_path: DB 저장 경로
    """
    print("\n" + "=" * 70)
    print("🔨 ChromaDB 구축 시작")
    print("=" * 70)
    
    # DB 디렉토리 생성
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_full_path = os.path.join(project_root, "data", "vectordb")
    os.makedirs(db_full_path, exist_ok=True)
    
    print(f"📁 DB 저장 경로: {db_full_path}")
    
    # ChromaDB 클라이언트 초기화
    chroma_client = chromadb.PersistentClient(path=db_full_path)
    
    # 기존 컬렉션 삭제 (있다면)
    try:
        chroma_client.delete_collection(name="youth_policies")
        print("🗑️  기존 컬렉션 삭제")
    except:
        pass
    
    # 새 컬렉션 생성
    collection = chroma_client.create_collection(
        name="youth_policies",
        metadata={"description": "온통청년 정책 데이터"}
    )
    
    print(f"\n📊 총 {len(policies)}개 정책 처리 중...")
    
    # 배치 처리를 위한 변수
    batch_size = 100
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for idx, policy in enumerate(policies, 1):
        # 정책 텍스트 생성
        policy_text = create_policy_text(policy)
        
        # 임베딩 생성
        try:
            embedding = get_embedding(policy_text)
            
            # 데이터 준비
            documents.append(policy_text)
            metadatas.append({
                '정책명': policy.get('정책명', ''),
                '대분류': policy.get('대분류', ''),
                '중분류': policy.get('중분류', ''),
                '주관기관명': policy.get('주관기관명', ''),
                '신청URL': policy.get('신청URL', ''),
                '정책키워드': policy.get('정책키워드', ''),
                # 신청 관련
                '신청기간': policy.get('신청기간', ''),
                '신청방법': policy.get('신청방법', ''),
                '제출서류': policy.get('제출서류', ''),
                # 자격 관련
                '추가자격조건': policy.get('추가자격조건', ''),
                '참여제외대상': policy.get('참여제외대상', ''),
                '지원최소연령': policy.get('지원최소연령', '0'),
                '지원최대연령': policy.get('지원최대연령', '0'),
                # 지원금 관련
                '최소지원금액': policy.get('최소지원금액', '0'),
                '최대지원금액': policy.get('최대지원금액', '0'),
            })
            ids.append(f"policy_{idx}")
            embeddings.append(embedding)
            
            # 진행상황 출력
            if idx % 10 == 0:
                print(f"  처리 중: {idx}/{len(policies)} ({idx/len(policies)*100:.1f}%)")
            
            # 배치 단위로 저장
            if len(documents) >= batch_size:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                print(f"  💾 배치 저장: {len(documents)}개")
                documents = []
                metadatas = []
                ids = []
                embeddings = []
                
        except Exception as e:
            print(f"  ⚠️  정책 {idx} 처리 오류: {e}")
            continue
    
    # 남은 데이터 저장
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"  💾 마지막 배치 저장: {len(documents)}개")
    
    print("\n" + "=" * 70)
    print("✅ ChromaDB 구축 완료!")
    print("=" * 70)
    print(f"📍 저장 위치: {db_full_path}")
    print(f"📊 총 저장된 정책 수: {collection.count()}")
    
    return collection


def test_search(collection, query="취업 지원 정책", top_k=3):
    """
    벡터 DB 검색 테스트
    
    Args:
        collection: ChromaDB 컬렉션
        query: 검색 쿼리
        top_k: 반환할 결과 수
    """
    print("\n" + "=" * 70)
    print("🔍 검색 테스트")
    print("=" * 70)
    print(f"질문: {query}\n")
    
    # 쿼리 임베딩 생성
    query_embedding = get_embedding(query)
    
    # 유사 문서 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    print(f"상위 {top_k}개 검색 결과:\n")
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0]), 1):
        print(f"[{i}] {metadata.get('정책명', 'N/A')}")
        print(f"    분야: {metadata.get('중분류', 'N/A')}")
        print(f"    담당: {metadata.get('주관기관명', 'N/A')}")
        print(f"    내용: {doc[:100]}...")
        print()


def main():
    print("=" * 70)
    print("ChromaDB 벡터 데이터베이스 구축")
    print("=" * 70)
    
    # API 키 확인
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 추가해주세요.")
        return
    
    print(f"✅ OpenAI API 키 설정 완료")
    
    # 전처리된 데이터 로드
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_path = os.path.join(project_root, "data", "processed", "youth_policies_filtered_kr_revised.json")
    
    if not os.path.exists(data_path):
        print(f"❌ 전처리된 데이터를 찾을 수 없습니다: {data_path}")
        return
    
    policies = load_preprocessed_data(data_path)
    
    # 샘플로 일부만 처리 (테스트용)
    # policies = policies[:50]  # 처음 50개만 테스트
    # 전체 데이터 사용
    print(f"⚠️  전체 {len(policies)}개 정책 처리 - 시간이 걸릴 수 있습니다.")
    
    # ChromaDB 구축
    collection = build_chromadb(policies)
    
    # 검색 테스트
    test_search(collection, "취업 지원 프로그램이 있나요?")
    test_search(collection, "창업 관련 정책을 알려주세요")
    
    print("\n✅ 모든 작업 완료!")


if __name__ == "__main__":
    main()
