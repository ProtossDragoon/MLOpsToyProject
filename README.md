# MLOps Toy Project

## 주제

스팸 문자 메시지 분류를 위한 MLOps 파이프라인 구축

## 목적

![image](https://github.com/ProtossDragoon/MLOpsToyProject/blob/main/docs/img/mlops_level1.svg)
이미지 출처: [Google Cloud](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning?hl=ko#mlops_level_0_manual_process)

- 다음 과정을 포함하는 Level 1 이상의 머신러닝 자동화 파이프라인을 구축합니다.
    - Serving
    - Experiment
    - Model Management 
    - Data Validation
    - Continuous Training
    - Monitoring
- [Kaggle SMS SPAM 데이터셋](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)을 사용합니다.
- [Full Stack Deep Learning course, Spring 2021](https://fullstackdeeplearning.com/spring2021/)를 학습합니다.

## 환경

`python >= 3.8.9` 의 가상환경 사용을 권장합니다.

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

로컬 개발 환경이 vscode 인 경우,
```bash
python3 -m ipykernel install --user --name MLOpsProject --display-name MLOpsProject
```

## 디렉터리

```text
├── docs: 프로젝트 문서 및 README 이미지들이 저장되어 있습니다.
│   └── ...
├── data
│   ├── sample: 샘플 데이터가 들어 있는 디렉터리
│   └── ...
├── notebooks
│   ├── ttf: notebook 환경에서 한글 시각화 지원을 위해 글꼴 설치 파일이 포함되어 있는 디렉터리
│   ├── eda.ipynb: EDA 노트북
│   └── ...
├── results
│   └── ...
├── src: 모델 학습, 모델 평가, 파이프라인, API 등 파일이 포함되어 있는 디렉터리
│   ├── app
│   │   ├── streamlit_app.py: streamlit 어플리케이션 파일
│   │   └── ...
│   └── ...
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

*placeholder 파일은 [이 글을 참고하여](https://mlops-guide.github.io/Structure/project_structure/) 디렉터리를 잡아주기 위한 파일로 아무 의미가 없습니다.
