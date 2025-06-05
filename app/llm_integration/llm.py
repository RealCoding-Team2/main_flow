import openai
import os
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일에서 환경변수 로드
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

class LLMRequester:
    def __init__(self, api_key: str = None, model: str = "gpt-4.1-mini", base_url: str = None):
        """
        LLM 모델과 소통하기 위한 요청 클래스
        
        Args:
            api_key: OpenAI API 키 (없으면 .env에서 OPENAI_API_KEY 사용)
            model: 사용할 모델명
            base_url: OpenAI API Base URL (없으면 .env에서 OPENAI_BASE_URL 사용)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        
        if not self.api_key:
            raise ValueError("OpenAI API key가 필요합니다. .env 파일에 OPENAI_API_KEY를 설정하거나 api_key 파라미터를 제공해주세요.")
        
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.conversation_history = []
        
        # 설정된 환경변수 정보 출력 (디버깅용)
        print(f"🤖 LLM 클라이언트 초기화 완료")
        print(f"   - Model: {self.model}")
        print(f"   - API Key: {'✅ 설정됨' if self.api_key else '❌ 미설정'}")
        print(f"   - Base URL: {self.base_url if self.base_url else '기본값 사용'}")
    
    def send_message(self, message: str, system_prompt: str = None, **kwargs) -> str:
        """
        LLM에게 메시지를 보내고 응답을 받습니다.
        
        Args:
            message: 사용자 메시지
            system_prompt: 시스템 프롬프트 (선택사항)
            **kwargs: 추가 OpenAI API 파라미터 (temperature, max_tokens 등)
            
        Returns:
            모델의 응답 텍스트
        """
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": message})
            
            # 기본값 설정
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", int(os.getenv("MAX_TOKENS", 1000))),
                "temperature": kwargs.get("temperature", float(os.getenv("TEMPERATURE", 0.7))),
                "top_p": kwargs.get("top_p", 1.0),
                "frequency_penalty": kwargs.get("frequency_penalty", 0),
                "presence_penalty": kwargs.get("presence_penalty", 0)
            }
            
            response = self.client.chat.completions.create(**api_params)
            
            assistant_response = response.choices[0].message.content
            
            self.conversation_history.append({"role": "user", "content": message})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"
    
    def clear_history(self):
        """대화 히스토리를 초기화합니다."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """현재 대화 히스토리를 반환합니다."""
        return self.conversation_history.copy()
    
    def change_model(self, model: str):
        """사용할 모델을 변경합니다."""
        self.model = model
        print(f"🔄 모델 변경: {model}")
    
    def change_base_url(self, base_url: str):
        """Base URL을 변경하고 클라이언트를 재생성합니다."""
        self.base_url = base_url
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)
        print(f"🔄 Base URL 변경: {base_url}")
    
    def get_env_info(self) -> dict:
        """현재 환경변수 정보를 반환합니다."""
        return {
            "api_key_set": bool(self.api_key),
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": int(os.getenv("MAX_TOKENS", 1000)),
            "temperature": float(os.getenv("TEMPERATURE", 0.7)),
        }


def load_env_config():
    """
    .env 파일에서 설정을 로드하고 검증합니다.
    
    Returns:
        dict: 로드된 환경변수 설정
    """
    load_dotenv()
    
    config = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL"),
        "DEFAULT_MODEL": os.getenv("DEFAULT_MODEL", "gpt-4.1-mini"),
    }
    
    return config


def simple_chat_example():
    """기본적인 채팅 예시"""
    print("=== LLM 간단한 채팅 예시 ===")
    
    try:
        # 환경변수에서 기본 모델 가져오기
        default_model = os.getenv("DEFAULT_MODEL", "gpt-4.1-mini")
        llm = LLMRequester(model=default_model)
        
        system_prompt = "당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정확한 답변을 제공해주세요."
        
        response = llm.send_message(
            "안녕하세요! 파이썬에서 리스트와 튜플의 차이점을 알려주세요.",
            system_prompt=system_prompt
        )
        
        print(f"\n사용자: 안녕하세요! 파이썬에서 리스트와 튜플의 차이점을 알려주세요.")
        print(f"LLM: {response}\n")
        
        response2 = llm.send_message("그럼 언제 리스트를 사용하고 언제 튜플을 사용하는 것이 좋을까요?")
        print(f"사용자: 그럼 언제 리스트를 사용하고 언제 튜플을 사용하는 것이 좋을까요?")
        print(f"LLM: {response2}")
        
    except ValueError as e:
        print(f"❌ 설정 오류: {e}")
        print("💡 .env 파일을 생성하고 OPENAI_API_KEY를 설정해주세요.")


if __name__ == "__main__":
    print("🔧 환경변수 설정 확인...")
    
    # 환경변수 설정 확인
    config = load_env_config()
        
        simple_chat_example() 