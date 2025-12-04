"""
청년 정책 RAG Pipeline
단계별로 구축하는 고급 RAG 시스템
"""

import os
from dotenv import load_dotenv
import chromadb
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document

# 환경 변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


class SimpleEnsembleRetriever:
    """간단한 Ensemble Retriever 구현 (Vector + BM25)"""
    
    def __init__(self, vector_retriever, bm25_retriever, embeddings):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.embeddings = embeddings
    
    def get_relevant_documents(self, query):
        """Vector와 BM25 결과를 결합"""
        # Vector 검색 (invoke 메서드 사용)
        try:
            vector_docs = self.vector_retriever.invoke(query)
        except:
            vector_docs = self.vector_retriever.get_relevant_documents(query)
        
        # BM25 검색
        try:
            bm25_docs = self.bm25_retriever.invoke(query)
        except:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # 결합 (중복 제거)
        seen_ids = set()
        combined_docs = []
        
        # Vector 결과 먼저 (50%)
        for doc in vector_docs[:5]:
            doc_id = doc.page_content[:100]  # 내용 앞부분으로 ID 대체
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)
        
        # BM25 결과 추가 (50%)
        for doc in bm25_docs[:5]:
            doc_id = doc.page_content[:100]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                combined_docs.append(doc)
        
        return combined_docs[:10]  # 상위 10개


class MultiQueryGenerator:
    """질문을 여러 관점으로 재작성하는 MultiQuery 생성기"""
    
    def __init__(self, llm):
        self.llm = llm
        self.prompt = self._create_prompt()
    
    def _create_prompt(self):
        """MultiQuery 프롬프트 생성"""
        template = """당신은 AI 검색 전문가입니다. 사용자의 질문을 다양한 관점에서 재작성하여 더 나은 검색 결과를 얻으려고 합니다.

원본 질문: {question}

위 질문을 **3가지 다른 방식**으로 재작성하세요:
1. 더 구체적으로
2. 더 넓은 관점에서
3. 다른 키워드 사용

응답 형식 (JSON):
{{
  "queries": [
    "재작성된 질문 1",
    "재작성된 질문 2",
    "재작성된 질문 3"
  ]
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def generate_queries(self, question):
        """질문을 여러 개로 확장"""
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": question})
            
            # JSON 파싱
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            queries = result.get("queries", [question])
            
            print(f"🔄 MultiQuery 생성: {len(queries)}개")
            for i, q in enumerate(queries, 1):
                print(f"  {i}. {q}")
            
            return queries
            
        except Exception as e:
            print(f"⚠️ MultiQuery 생성 실패: {e}, 원본 질문만 사용")
            return [question]


class YouthPolicyRAG:
    """청년 정책 RAG 시스템"""
    
    def __init__(self, db_path="../data/vectordb"):
        """
        초기화
        
        Args:
            db_path: ChromaDB 경로
        """
        print("🚀 RAG Pipeline 초기화 중...")
        
        # LLM 초기화
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=OPENAI_API_KEY
        )
        
        # 임베딩 모델
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=OPENAI_API_KEY
        )
        
        # Vector Store 로드
        current_dir = os.path.dirname(os.path.abspath(__file__))
        full_db_path = os.path.join(current_dir, db_path)
        
        self.vectorstore = Chroma(
            persist_directory=full_db_path,
            collection_name="youth_policies",
            embedding_function=self.embeddings
        )
        
        # ChromaDB collection 직접 접근 (필터링용)
        chroma_client = chromadb.PersistentClient(path=full_db_path)
        self.collection = chroma_client.get_collection(name="youth_policies")
        
        # BM25 Retriever 초기화 (키워드 기반 검색)
        self._init_bm25_retriever()
        
        # Ensemble Retriever 생성 (Vector + BM25)
        self._init_ensemble_retriever()
        
        # MultiQuery Generator 초기화
        self.multi_query_gen = MultiQueryGenerator(self.llm)
        
        # 사용자 정보 (나이, 지역)
        self.user_age = None
        self.user_region = None
        
        # MultiQuery 사용 여부 (기본: True)
        self.use_multi_query = True
        
        # Retriever 생성 (기본)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # 프롬프트 템플릿
        self.prompt = self._create_prompt()
        
        # Router 프롬프트
        self.router_prompt = self._create_router_prompt()
        
        # RAG 체인 구성
        self.rag_chain = self._build_chain()
        
        print("✅ RAG Pipeline 초기화 완료!")
    
    def _init_bm25_retriever(self):
        """BM25 Retriever 초기화 (키워드 기반 검색)"""
        print("📚 BM25 Retriever 초기화 중...")
        
        # ChromaDB에서 모든 문서 가져오기
        all_data = self.collection.get()
        
        # Document 객체로 변환
        documents = []
        for doc_text, metadata in zip(all_data['documents'], all_data['metadatas']):
            documents.append(Document(
                page_content=doc_text,
                metadata=metadata
            ))
        
        # BM25 Retriever 생성
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 10  # 상위 10개 검색
        
        print(f"✅ BM25 Retriever 초기화 완료 (문서 수: {len(documents)}개)")
    
    def _init_ensemble_retriever(self):
        """Ensemble Retriever 초기화 (Vector + BM25)"""
        print("🔗 Ensemble Retriever 생성 중...")
        
        # Vector Retriever (의미 기반)
        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        # 간단한 Ensemble 구현 (직접 구현)
        self.ensemble_retriever = SimpleEnsembleRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=self.bm25_retriever,
            embeddings=self.embeddings
        )
        print("✅ Ensemble Retriever 생성 완료 (Vector + BM25)")
    
    def _create_router_prompt(self):
        """Router 프롬프트 생성"""
        template = """당신은 질문을 분석하여 적절한 작업을 선택하는 라우터입니다.

질문: {question}

다음 중 하나를 선택하세요:

1. SEARCH_POLICY
   - 청년 정책 검색이 필요한 경우
   - 예: "창업 지원금", "취업 지원", "주거 지원", "대출", "교육" 등

2. GENERAL_CHAT
   - 일반적인 인사, 감사 표현
   - 예: "안녕하세요", "고맙습니다", "도움이 되었어요"

3. REQUEST_INFO
   - 사용자 정보(나이, 지역)가 필요한 경우
   - 예: 정책 질문인데 사용자 정보가 없는 경우

4. CLARIFY
   - 질문이 불명확하여 추가 정보가 필요한 경우
   - 예: "정책", "지원금" 같이 너무 광범위한 질문

**중요**: 반드시 JSON 형식으로만 답변하세요.

응답 형식:
{{
  "action": "SEARCH_POLICY",
  "reason": "창업 지원금 관련 정책 검색 필요",
  "keywords": ["창업", "지원금"]
}}

답변:"""
        return ChatPromptTemplate.from_template(template)
    
    def _create_prompt(self):
        """프롬프트 템플릿 생성"""
        template = """당신은 청년 정책 전문 상담사입니다.
사용자의 질문에 대해 제공된 정책 정보를 바탕으로 친절하고 정확하게 답변하세요.

📋 정책 정보:
{context}

❓ 사용자 질문:
{question}

💡 답변 가이드라인:
1. 제공된 정책 정보만 사용하세요
2. 정책명, 지원내용, 신청방법을 명확히 설명하세요
3. 정보가 부족하면 "제공된 정보에는 없습니다"라고 말하세요
4. 친근하고 격려하는 톤으로 작성하세요
5. 필요시 추가 질문을 유도하세요

답변:"""
        
        return ChatPromptTemplate.from_template(template)
    
    def _build_chain(self):
        """RAG 체인 구성"""
        chain = (
            {
                "context": RunnableLambda(self._retrieve_and_filter) | RunnableLambda(self._format_docs),
                "question": RunnablePassthrough()
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def _retrieve_and_filter(self, question):
        """검색 + 메타데이터 필터링 (MultiQuery + Ensemble 사용)"""
        
        # MultiQuery: 질문을 여러 개로 확장
        if self.use_multi_query:
            queries = self.multi_query_gen.generate_queries(question)
        else:
            queries = [question]
        
        # 모든 쿼리로 검색 후 결과 통합
        all_docs = []
        seen_ids = set()
        
        for query in queries:
            try:
                # Ensemble에서 검색
                docs = self.ensemble_retriever.get_relevant_documents(query)
                
                # 중복 제거하면서 추가
                for doc in docs:
                    doc_id = doc.page_content[:100]
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                        
            except Exception as e:
                print(f"⚠️ 쿼리 '{query}' 검색 오류: {e}")
                continue
        
        print(f"🔍 총 검색 결과: {len(all_docs)}개 (중복 제거)")
        
        # 사용자 정보가 없으면 그대로 반환
        if not (self.user_age or self.user_region):
            return all_docs[:5]
        
        # 필터링 시작
        filtered_docs = []
        for doc in all_docs:
            metadata = doc.metadata
            
            # 나이 필터링
            age_match = True
            if self.user_age:
                try:
                    min_age = int(metadata.get('지원최소연령', '0') or '0')
                    max_age = int(metadata.get('지원최대연령', '0') or '0')
                    
                    if min_age > 0 and self.user_age < min_age:
                        age_match = False
                    if max_age > 0 and max_age < 999 and self.user_age > max_age:
                        age_match = False
                except:
                    pass
            
            # 지역 필터링 (계층적 매칭: 전국 → 시/도 → 시/군/구)
            region_match = True
            if self.user_region:
                org_name = metadata.get('주관기관명', '')
                additional_cond = metadata.get('추가자격조건', '')
                reg_group = metadata.get('재공기관그룹', '')
                
                policy_name = metadata.get('정책명', 'N/A')
                
                # 1순위: 전국 정책은 항상 포함
                if '중앙부처' in reg_group or '전국' in org_name:
                    region_match = True
                    print(f"  ✓ 전국 정책: {policy_name} (기관: {org_name})")
                else:
                    # 2순위: 시/도 단위 매칭 (구/군 입력 시에도 시/도 정책 포함)
                    sido_list = ['서울', '경기', '인천', '부산', '대구', '광주', '대전', '울산', '세종',
                               '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
                    
                    user_sido = None
                    for sido in sido_list:
                        if sido in self.user_region:
                            user_sido = sido
                            break
                    
                    # 시/도 매칭 확인
                    if user_sido and user_sido in org_name:
                        region_match = True
                        print(f"  ✓ 시/도 매칭: {policy_name} (시/도: {user_sido}, 기관: {org_name})")
                    else:
                        # 3순위: 구/군 단위 상세 매칭
                        region_clean = self.user_region.replace('특별시', '').replace('광역시', '').replace('특별자치시', '')
                        region_clean = region_clean.replace('도', '').replace('시', '').replace('군', '').replace('구', '').strip()
                        
                        user_region_tokens = []
                        if user_sido:
                            user_region_tokens.append(user_sido)
                        
                        for token in region_clean.split():
                            if token and token not in user_region_tokens:
                                user_region_tokens.append(token)
                        
                        region_match = False
                        for token in user_region_tokens:
                            if token in org_name or token in additional_cond:
                                region_match = True
                                print(f"  ✓ 상세 매칭: {policy_name} (토큰: {token}, 기관: {org_name})")
                                break
                        
                        if not region_match:
                            print(f"  ✗ 제외: {policy_name} (기관: {org_name})")
            
            # 두 조건 모두 만족하면 포함
            if age_match and region_match:
                filtered_docs.append(doc)
        
        print(f"✅ 필터링 후: {len(filtered_docs)}개")
        
        # 결과가 너무 적으면 전국 정책만이라도 반환
        if len(filtered_docs) < 3:
            print("⚠️ 필터링 결과 부족, 전국 정책 추가 검색")
            for doc in all_docs:
                if len(filtered_docs) >= 5:
                    break
                metadata = doc.metadata
                reg_group = metadata.get('재공기관그룹', '')
                if '중앙부처' in reg_group and doc not in filtered_docs:
                    filtered_docs.append(doc)
        
        return filtered_docs[:5]
    
    def _fallback_retrieve(self, question):
        """폴백: 기존 ChromaDB 직접 검색"""
        query_embedding = self.embeddings.embed_query(question)
        search_count = 25 if (self.user_age or self.user_region) else 5
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=search_count
        )
        
        if not results['documents'][0]:
            return []
        
        filtered_docs = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            filtered_docs.append(Document(
                page_content=doc,
                metadata=metadata
            ))
        
        return filtered_docs[:5]
    
    def _format_docs(self, docs):
        """문서 포맷팅"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.metadata
            formatted.append(f"""
[정책 {i}]
정책명: {metadata.get('정책명', 'N/A')}
분야: {metadata.get('대분류', 'N/A')} > {metadata.get('중분류', 'N/A')}
담당기관: {metadata.get('주관기관명', 'N/A')}
연령: {metadata.get('지원최소연령', 'N/A')}세 ~ {metadata.get('지원최대연령', 'N/A')}세
지원금액: {metadata.get('최소지원금액', '0')}원 ~ {metadata.get('최대지원금액', '0')}원
내용: {doc.page_content[:500]}...
""")
        return "\n".join(formatted)
    
    def query(self, question: str):
        """
        질문에 답변 (Router 적용)
        
        Args:
            question: 사용자 질문
            
        Returns:
            str: 답변
        """
        user_info = ""
        if self.user_age or self.user_region:
            user_info = f" (나이: {self.user_age}세, 지역: {self.user_region})"
        
        print(f"\n🔍 질문: {question}{user_info}")
        
        # 1단계: Router로 질문 분석
        routing_result = self.route_query(question)
        action = routing_result.get('action')
        
        # 2단계: Action에 따라 처리
        if action == "GENERAL_CHAT":
            # 일반 대화 - 검색 없이 직접 응답
            print("💬 일반 대화 모드\n")
            chat_prompt = ChatPromptTemplate.from_template(
                "당신은 친근한 청년 정책 상담사입니다. 다음 질문에 간단히 답변하세요.\n\n질문: {question}\n\n답변:"
            )
            response = (chat_prompt | self.llm | StrOutputParser()).invoke({"question": question})
            return response
        
        elif action == "REQUEST_INFO":
            # 사용자 정보 요청
            print("📝 사용자 정보 필요\n")
            return """더 정확한 정책을 추천해드리기 위해 정보가 필요합니다! 😊

다음 정보를 알려주시겠어요?
1. 나이: 만 몇 세이신가요?
2. 지역: 어디에 거주하시나요? (예: 서울특별시, 경기도 의정부시)

정보를 입력하시면 맞춤형 정책을 찾아드리겠습니다!"""
        
        elif action == "CLARIFY":
            # 질문 명확화 요청
            print("❓ 질문 명확화 필요\n")
            return """질문을 좀 더 구체적으로 말씀해주시겠어요? 😊

예를 들면:
- "창업 지원금이 궁금해요"
- "청년 취업 지원 프로그램 알려주세요"
- "전월세 대출 정책이 있나요?"

구체적인 분야를 말씀해주시면 더 정확한 정책을 찾아드릴게요!"""
        
        else:  # SEARCH_POLICY
            # 정책 검색 - RAG 체인 실행
            print("⏳ 정책 검색 중...\n")
            response = self.rag_chain.invoke(question)
            return response
    
    def set_user_info(self, age=None, region=None):
        """
        사용자 정보 설정
        
        Args:
            age: 나이
            region: 지역 (예: "경기도 의정부시")
        """
        self.user_age = age
        self.user_region = region
        
        info = []
        if age:
            info.append(f"나이 {age}세")
        if region:
            info.append(f"지역 {region}")
        
        if info:
            print(f"✅ 사용자 정보 설정: {', '.join(info)}")
            print(f"   → 전국/중앙부처 정책 + {region} 정책이 함께 검색됩니다.")
    
    def route_query(self, question: str):
        """
        질문을 분석하여 적절한 작업으로 라우팅
        
        Args:
            question: 사용자 질문
            
        Returns:
            dict: 라우팅 결과
        """
        try:
            # Router LLM 호출
            router_chain = self.router_prompt | self.llm | StrOutputParser()
            response = router_chain.invoke({"question": question})
            
            # JSON 파싱
            # 응답에서 JSON 부분만 추출 (```json...``` 제거)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response)
            
            # REQUEST_INFO인 경우, 사용자 정보가 이미 있으면 SEARCH_POLICY로 변경
            if result.get('action') == 'REQUEST_INFO':
                if self.user_age or self.user_region:
                    print(f"ℹ️  사용자 정보 이미 있음 (나이: {self.user_age}, 지역: {self.user_region})")
                    result['action'] = 'SEARCH_POLICY'
                    result['reason'] = '사용자 정보 있음, 정책 검색 진행'
            
            print(f"🎯 라우팅 결과: {result['action']} - {result.get('reason', '')}")
            
            return result
            
        except Exception as e:
            print(f"⚠️ 라우팅 오류: {e}, 기본 검색으로 진행")
            return {
                "action": "SEARCH_POLICY",
                "reason": "라우팅 실패, 기본 검색",
                "keywords": []
            }
    
    def interactive_mode(self):
        """대화형 모드"""
        print("\n" + "=" * 70)
        print("💬 청년 정책 상담 챗봇")
        print("=" * 70)
        
        # 사용자 정보 입력
        print("\n👤 사용자 정보를 입력해주세요 (Enter로 건너뛰기 가능)")
        
        age_input = input("나이: ").strip()
        if age_input:
            try:
                self.user_age = int(age_input)
            except:
                print("⚠️ 유효하지 않은 나이입니다.")
        
        region_input = input("지역 (예: 경기도 의정부시, 서울특별시): ").strip()
        if region_input:
            self.user_region = region_input
        
        if self.user_age or self.user_region:
            self.set_user_info(self.user_age, self.user_region)
        
        print("\n질문을 입력하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")
        
        while True:
            try:
                question = input("👤 질문: ").strip()
                
                if question.lower() in ['quit', 'exit', '종료', 'q']:
                    print("\n👋 상담을 종료합니다. 감사합니다!")
                    break
                
                if not question:
                    continue
                
                # 답변 생성
                answer = self.query(question)
                print(f"\n🤖 답변:\n{answer}\n")
                print("-" * 70)
                
            except KeyboardInterrupt:
                print("\n\n👋 상담을 종료합니다.")
                break
            except Exception as e:
                print(f"\n❌ 오류 발생: {e}\n")


def main():
    """메인 함수"""
    # RAG 시스템 초기화
    rag = YouthPolicyRAG()
    
    # 사용자 정보 설정 (테스트용)
    # rag.set_user_info(age=27, region="경기도 의정부시")
    
    # 테스트 질문
    print("\n" + "=" * 70)
    print("🧪 Router 테스트")
    print("=" * 70)
    
    # 대화형 모드
    print("\n대화형 모드로 전환합니다...\n")
    rag.interactive_mode()


if __name__ == "__main__":
    main()
