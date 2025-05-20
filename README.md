<img src="./assets/cover.png" style="width: 100%" />

## WhatLm

### 🎓 모델

-   `v0` - 텍스트 생성만 가능한 초기 버전
-   `qna-alpha` - 간단히 구현한 qna 모델 알파버전
-   `v0-1` - **v0** 모델에서 더욱 개선된 텍스트 생성만 가능한 버전
-   `v0-2` - **v0-1** 에서 extend 시킨 qna 가능한 버전 (HuggingFace 사용)
-   `v0-3` (개발 중단) - AI 도움 없이 직접 작성한 마르코프 체인
-   `v0-4` - 트랜스포머 아키텍처, 멀티헤드 어텐션, 셀프 어텐션, 포지셔널 인코딩, 피드포워드 네트워킹, 레이어 정규화, GELU 활성화 함수, 소프트맥스 함수, 크로스 엔트로피 손실 함수, Adam 옵티마이저, Top-K 샘플링 등 다양한 알고리즘을 사용한 상위 버전

### 📚 클론

```bash
$ git clone https://github.com/inf-hubof/whatlm.git
```

### ⚡️ 설치

```bash
$ npm i
```

### 🚀 실행

> v0-4의 경우엔 index.ts 파일의 실행 부분을 수정해 주세요!

```bash
$ npx ts-node models/<모델 이름>

# v0-2의 경우
$ npx ts-node models/v0-2/test
```
