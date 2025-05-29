# 기본 템플릿
- 서버주소로 접속 시 `프론트엔드 페이지` 전송
- 클라이언트에서 `이미지` 전송 및 서버에서 수신
- <img src="/image_for_explain/client page demo2.gif" width="640px">

---
# 폴더 구조 ( `root` )
- `config.py`: 
  - `.env`설정파일에서 값들을 불러옵니다 
  - `flask`의 메인 객체에 설정 값들을 주입합니다.
- `run.py`: 
  - `python run.py`를 입력할 시 서버가 동작되도록 합니다
  - 설정된 라우터는 `/app/__init__.py`를 참고하세요
- `/app`: 핵심코드는 전부 여기에 있습니다
---
# 폴더 구조 ( `/app` )
- `/deepfake_detector`:
  - 딥페이크 탐지에 대한 모듈입니다
  - 탐지 요청(라우팅)도 여기서 처리합니다
- `/models`: 딥페이크 탐지에 사용되는 AI모델이 저장되있습니다
- `/static`, `/templates`: `클라이언트(Frontend)`에 대한 코드가 저장되있습니다
- `__init__.py`:
  - `flask`의 메인 객체를 실제로 생성 및 초기화합니다
  - 각 모듈(`/deepfake_detector`, `/(LLM관련 모듈)`)에 대한 라우트(블루프린트)를 등록합니다
