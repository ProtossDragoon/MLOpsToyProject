# Backend

이 디렉터리의 소스코드는 오픈소스 백엔드 프레임워크 [FastAPI](https://fastapi.tiangolo.com/) 를 사용합니다.

## 실행

API 서버가 잘 동작하는지 확인하기 위해 일부 기능을 실행시켜볼 수 있습니다. **프로젝트 루트**에서 다음 명령을 실행합니다.

```bash
uvicorn src.backend.app.main:app --reload
```

문제없이 명령어 실행이 완료되었다면 [로컬 서버](http://127.0.0.1:8000)에 접속합니다. [API 명세](http://127.0.0.1:8000/docs)를 살펴보고 동작을 직접 확인해볼 수 있습니다.

## 디렉터리

*placeholder 파일은 [FastAPI 프로젝트 리포지토리](https://github.com/tiangolo/full-stack-fastapi-postgresql/tree/master/%7B%7Bcookiecutter.project_slug%7D%7D/backend/app)를 참고하여 디렉터리를 잡아주기 위한 파일로 아무 의미가 없습니다.
