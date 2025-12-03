"""
온통청년 API를 통한 정책 데이터 수집
"""

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()
YOUTH_POLICY_API = os.getenv('YOUTH_POLICY_API')

def fetch_youth_policies(page_size):
    """
    온통청년 API를 통해 정책 데이터 가져오기
    
    Args:
        page_size (int): 가져올 정책 개수
    
    Returns:
        dict: API 응답 데이터
    """
    # 여러 가능한 엔드포인트 시도
    endpoints = [
        {
            'url': "https://www.youthcenter.go.kr/go/ythip/getPlcy",
            'params': {
                'apiKeyNm': YOUTH_POLICY_API,
                'pageSize': page_size,
            }
        }
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
    }
    
    print(f"API Key 설정 여부: {'✅' if YOUTH_POLICY_API else '❌'}")
    
    for i, endpoint in enumerate(endpoints, 1):
        api_url = endpoint['url']
        params = endpoint['params']
        
        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=60)
            
            
            if response.status_code == 200:
                print(f"✅ 응답 성공!")
                print(f"응답 크기: {len(response.text):,} bytes")
                
                # JSON 파싱
                try:
                    data = response.json()
                    return data
                    
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 파싱 오류: {e}")
                    print(f"응답 내용 일부: {response.text[:500]}")
                    continue
            else:
                print(f"⚠️  상태 코드 {response.status_code} - 다음 엔드포인트 시도...")
                print(f"응답 내용: {response.text[:200]}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 오류: {e}")
            continue
    
    print("\n❌ 모든 엔드포인트 시도 실패")
    return None


def save_json(data, filename="youth_policies_api"):
    """
    데이터를 JSON 파일로 저장
    """
    # 현재 스크립트 위치에서 data/raw 경로 계산
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(raw_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(raw_dir, f"{filename}.json")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 저장 완료: {filepath}")
    
    # 파일 크기 확인
    file_size = os.path.getsize(filepath)
    print(f"파일 크기: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    
    return filepath


def main():
    print("=" * 70)
    print("온통청년 정책 API 데이터 수집")
    print("=" * 70)
    
    # API 호출
    data = fetch_youth_policies(page_size=3000)
    
    if data:
        # JSON 저장
        filepath = save_json(data)
        
        print("\n" + "=" * 70)
        print("✅ 완료!")
        print("=" * 70)
        
        # 통계 정보
        if isinstance(data, dict):
            total_count = 0
            for key, value in data.items():
                if isinstance(value, list):
                    count = len(value)
                    total_count += count
                    print(f"  {key}: {count}개")
            if total_count > 0:
                print(f"\n  총 정책 수: {total_count}개")
        elif isinstance(data, list):
            print(f"  총 정책 수: {len(data)}개")
        
    else:
        print("\n❌ 데이터 수집 실패")


if __name__ == "__main__":
    main()
