"""
ChromaDB 검증 및 테스트 스크립트
구축된 벡터 DB의 상태를 확인하고 다양한 쿼리로 테스트
"""

import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)


def get_embedding(text, model="text-embedding-3-small"):
    """텍스트 임베딩 생성"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def load_chromadb():
    """ChromaDB 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    db_path = os.path.join(project_root, "data", "vectordb")
    
    if not os.path.exists(db_path):
        print(f"❌ ChromaDB를 찾을 수 없습니다: {db_path}")
        return None
    
    print(f"📂 DB 경로: {db_path}")
    
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = chroma_client.get_collection(name="youth_policies")
        return collection
    except Exception as e:
        print(f"❌ 컬렉션을 찾을 수 없습니다: {e}")
        return None


def check_db_stats(collection):
    """DB 통계 확인"""
    print("\n" + "=" * 70)
    print("📊 ChromaDB 통계")
    print("=" * 70)
    
    count = collection.count()
    print(f"✅ 저장된 정책 수: {count}개")
    
    # 샘플 데이터 확인
    sample = collection.peek(limit=3)
    
    print(f"\n📄 샘플 데이터 (3개):")
    print("-" * 70)
    
    for i, (id, doc, metadata) in enumerate(zip(sample['ids'], sample['documents'], sample['metadatas']), 1):
        print(f"\n[{i}] ID: {id}")
        print(f"    정책명: {metadata.get('정책명', 'N/A')}")
        print(f"    분야: {metadata.get('중분류', 'N/A')}")
        print(f"    담당: {metadata.get('주관기관명', 'N/A')}")
        print(f"    내용: {doc[:150]}...")
    
    return count


def test_search(collection, query, top_k=5, user_info=None):
    """검색 테스트
    
    Args:
        collection: ChromaDB 컬렉션
        query: 검색 질문
        top_k: 반환할 결과 수
        user_info: 사용자 정보 딕셔너리 {'age': 27, 'region': '경기'}
    """
    print("\n" + "=" * 70)
    print("🔍 검색 테스트")
    print("=" * 70)
    print(f"질문: {query}")
    if user_info:
        print(f"👤 사용자 정보: 나이 {user_info.get('age', 'N/A')}세, 지역 {user_info.get('region', 'N/A')}")
    print(f"검색 결과 수: {top_k}개\n")
    
    # 쿼리 임베딩
    query_embedding = get_embedding(query)
    
    # 필터링을 위해 더 많은 결과 가져오기
    search_count = top_k * 5 if user_info else top_k
    
    # 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=search_count
    )
    
    if not results['documents'][0]:
        print("❌ 검색 결과가 없습니다.")
        return
    
    # 사용자 정보로 필터링
    filtered_results = []
    if user_info:
        user_age = user_info.get('age')
        user_region = user_info.get('region', '').strip()
        
        for doc, metadata, distance in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0] if 'distances' in results else [0] * len(results['documents'][0])
        ):
            # 나이 필터링
            age_match = True
            if user_age:
                try:
                    min_age_str = metadata.get('지원최소연령', '0') or '0'
                    max_age_str = metadata.get('지원최대연령', '0') or '0'
                    min_age = int(min_age_str)
                    max_age = int(max_age_str)
                    
                    # 연령 체크 (0은 제한 없음)
                    if min_age > 0 and user_age < min_age:
                        age_match = False
                    if max_age > 0 and max_age < 999 and user_age > max_age:
                        age_match = False
                except:
                    pass
            
            # 지역 필터링
            region_match = True
            if user_region:
                org_name = metadata.get('주관기관명', '')
                additional_cond = metadata.get('추가자격조건', '')
                
                # 전국 정책은 항상 포함
                if '전국' in org_name:
                    region_match = True
                else:
                    # 사용자 입력을 토큰으로 분리 (예: "경기도 의정부시" → ["경기", "의정부"])
                    user_region_tokens = []
                    # 시도 추출
                    sido_list = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                               '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
                    for sido in sido_list:
                        if sido in user_region:
                            user_region_tokens.append(sido)
                            break
                    
                    # 시군구 추출 (도/시/군/구 제거)
                    region_clean = user_region.replace('특별시', '').replace('광역시', '').replace('특별자치시', '')
                    region_clean = region_clean.replace('도', '').replace('시', '').replace('군', '').replace('구', '').strip()
                    
                    # 공백으로 분리된 나머지 지역명 추가
                    for token in region_clean.split():
                        if token and token not in user_region_tokens:
                            user_region_tokens.append(token)
                    
                    # 토큰 중 하나라도 매칭되면 OK
                    region_match = False
                    for token in user_region_tokens:
                        if token in org_name or token in additional_cond:
                            region_match = True
                            break
            
            # 두 조건 모두 만족하면 결과에 포함
            if age_match and region_match:
                filtered_results.append((doc, metadata, distance))
                if len(filtered_results) >= top_k:
                    break
        
        if not filtered_results:
            print(f"❌ 사용자 조건에 맞는 정책이 없습니다.")
            print(f"   (나이: {user_age}세, 지역: {user_region})")
            return
        
        print(f"✅ 필터링 후 {len(filtered_results)}개 결과 발견 (전체 {len(results['documents'][0])}개 중)\n")
        results_to_show = filtered_results
    else:
        print(f"✅ {len(results['documents'][0])}개 결과 발견\n")
        results_to_show = list(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0] if 'distances' in results else [0] * len(results['documents'][0])
        ))
    
    for i, (doc, metadata, distance) in enumerate(results_to_show, 1):
        # 연령 정보 처리
        min_age = metadata.get('지원최소연령', '0') or '0'
        max_age = metadata.get('지원최대연령', '0') or '0'
        try:
            min_age_int = int(min_age)
            max_age_int = int(max_age)
            if min_age_int == 0 and max_age_int == 0:
                age_info = "제한 없음"
            elif min_age_int == 0:
                age_info = f"~{max_age_int}세"
            elif max_age_int == 0 or max_age_int == 999:
                age_info = f"{min_age_int}세~"
            else:
                age_info = f"{min_age_int}세~{max_age_int}세"
        except:
            age_info = f"{min_age}~{max_age}"
        
        # 지역 정보 추출
        org_name = metadata.get('주관기관명', 'N/A')
        if '전국' in org_name:
            region_info = "🌏 전국"
        else:
            # 시도 정보 추출
            regions = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                      '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
            found_region = None
            for region in regions:
                if region in org_name:
                    found_region = region
                    break
            region_info = f"📍 {found_region}" if found_region else f"📍 {org_name}"
        
        print(f"[{i}] {metadata.get('정책명', 'N/A')}")
        print(f"    {region_info}")
        print(f"    👤 연령: {age_info}")
        print(f"    📂 분야: {metadata.get('중분류', 'N/A')}")
        print(f"    🏢 담당: {org_name}")
        print(f"    💰 지원금: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원")
        print(f"    📅 신청기간: {metadata.get('신청기간', 'N/A')}")
        print(f"    🔗 URL: {metadata.get('신청URL', 'N/A')}")
        print(f"    📏 유사도: {distance:.4f}")
        print(f"    📝 내용: {doc[:150]}...")
        print()


def interactive_search(collection):
    """대화형 검색"""
    print("\n" + "=" * 70)
    print("💬 대화형 검색 모드 (종료: 'quit', 'q', 'exit')")
    print("=" * 70)
    
    # 사용자 정보 입력
    print("\n👤 사용자 정보를 입력하세요 (선택사항, 엔터로 건너뛰기)")
    user_age_input = input("나이: ").strip()
    user_region_input = input("지역 (예: 서울, 경기, 부산): ").strip()
    
    user_info = {}
    if user_age_input:
        try:
            user_info['age'] = int(user_age_input)
        except:
            print("⚠️  나이를 숫자로 입력해주세요. 필터링 없이 진행합니다.")
    if user_region_input:
        user_info['region'] = user_region_input
    
    if user_info:
        print(f"\n✅ 사용자 정보 설정: 나이 {user_info.get('age', 'N/A')}세, 지역 {user_info.get('region', 'N/A')}")
    else:
        print("\n✅ 필터링 없이 검색합니다.")
    
    while True:
        try:
            query = input("\n질문을 입력하세요: ").strip()
            
            if query.lower() in ['quit', 'q', 'exit', '종료']:
                print("검색을 종료합니다.")
                break
            
            if not query:
                continue
            
            test_search(collection, query, top_k=5, user_info=user_info if user_info else None)
            
        except KeyboardInterrupt:
            print("\n\n검색을 종료합니다.")
            break
        except Exception as e:
            print(f"❌ 오류 발생: {e}")


def main():
    print("=" * 70)
    print("ChromaDB 검증 및 테스트")
    print("=" * 70)
    
    # DB 로드
    collection = load_chromadb()
    
    if not collection:
        return
    
    # 1. DB 통계 확인
    count = check_db_stats(collection)
    
    if count == 0:
        print("\n❌ DB가 비어있습니다. build_vectordb.py를 먼저 실행하세요.")
        return
    
    # 2. 미리 정의된 테스트 쿼리들
    test_queries = [
        "취업 지원 프로그램이 있나요?",
        "창업 관련 정책을 알려주세요",
        "청년 주거 지원 정책은?",
        "해외 취업이나 인턴십 프로그램",
        "교육 바우처 지원"
    ]
    
    print("\n" + "=" * 70)
    print("🧪 자동 테스트 쿼리")
    print("=" * 70)
    
    for query in test_queries:
        test_search(collection, query, top_k=3)
        input("\n[Enter]를 눌러 다음 테스트로 진행...")
    
    # 3. 대화형 검색
    print("\n" + "=" * 70)
    response = input("대화형 검색을 시작하시겠습니까? (y/n): ").strip().lower()
    
    if response in ['y', 'yes', 'ㅛ']:
        interactive_search(collection)
    
    print("\n✅ 검증 완료!")


if __name__ == "__main__":
    main()
