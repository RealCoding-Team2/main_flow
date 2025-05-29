# ❓ 서버 실행 방법
- `.env`파일 생성
  - 루트 디렉토리에 위치
  - ```
    # 파일 예시
    GEMINI_API_URL=<제미나이 API 키>
    ```
- 파이썬 버전: 3.13
- 패키지 설치: `pip install -r requirements.txt`
- dlib 설치:
  - CMake 설치 필요 ([설치 페이지](https://cmake.org/download/)) ([Direct 설치](https://github.com/Kitware/CMake/releases/download/v4.0.2/cmake-4.0.2-windows-x86_64.msi))
  - dlib 패키지 파일 설치 ([설치 페이지](https://dlib.net/)) ([Direct 설치](http://dlib.net/files/dlib-20.0.zip))
  - dlib CMake 빌드 ( 꼭 CMD 관리자 권한으로 실행 할 것! )
  - 참고 사이트: https://sulastri.tistory.com/3
- 실행: `python run.py`

---

# </> 기본 템플릿
- 서버주소로 접속 시 `프론트엔드 페이지` 전송
- 클라이언트에서 `이미지` 전송 및 서버에서 수신
- <img src="/image_for_explain/client page demo2.gif" width="640px">

---

# 📂 폴더 구조 ( `root` )
- `config.py`: 
  - `.env`설정파일에서 값들을 불러옵니다 
  - `flask`의 메인 객체에 설정 값들을 주입합니다.
- `run.py`: 
  - `python run.py`를 입력할 시 서버가 동작되도록 합니다
  - 설정된 라우터는 `/app/__init__.py`를 참고하세요
- `/app`: 핵심코드는 전부 여기에 있습니다

---

# 📚 참고자료
1. https://wikidocs.net/81504  // 적용한 디자인 패턴
2. https://sulastri.tistory.com/3  // dlib 설치 방법
