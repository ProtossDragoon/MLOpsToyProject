# Custom Models

이 디렉터리의 소스코드는 오픈소스 머신러닝 프레임워크 [TensorFlow](https://github.com/tensorflow/tensorflow) 를 사용합니다.

## 환경

`python >= 3.8.9` 의 가상환경 사용을 권장합니다.

### macOS

- [ ] Intel MacOS
- [x] Apple Silicon MacOS

```bash
python3 -m pip install --upgrade pip
```

MacOS 를 위한 TensorFlow 설치
```bash
python3 -m pip install tensorflow-macos
```

- **NOTE**: Apple Silicon GPU 가속을 사용하려면 Conda 환경을 사용해야 합니다. 이 문서에서는 GPU 가속이 가능한 tensorflow 를 설치하지 않습니다.

### Windows

`TODO`

### Linux

`TODO`

## 실행

모델링 및 학습이 잘 동작하는지 확인하기 위해 일부 기능을 실행시켜볼 수 있습니다. **프로젝트 루트**에서 다음 명령을 실행합니다. 이 명령은 샘플 데이터를 이용해 자동으로 전처리, 모델링, 학습, 모델 평가를 진행합니다.

```bash
python3 -m src.model.custom.models.mlp
python3 -m src.model.custom.models.lstm
```