# 개인화 와인 추천 시스템

## 문제 정의 
1. 와인 소비량이 증가하는 추세
2. 다만 초보자가 자신에게 맞는 와인을 찾기는 어려움
3. 이를 UI/UX적인 경험과 AI로 해결하고자 함

## 문제 해결 과정

### 콜드 스타트 문제
1. 일반적으로 추천시스템은 아이템-유저 간의 상호작용 정보가 필수적
2. 초기 유저는 이 정보가 제한적이며 초기 추천이 정확하지 않을 시에는 이탈 가능성이 높음 -> 이를 해결하기 위해 온보딩 과정이 수반됨
3. 다만 온보딩은 상세할수록 좋지만 유저 입장에서는 피곤하고 지루하여 이탈 가능성이 높음
4. 하지만 MBTI 테스트는 최소 10분이 걸림에도 많은 사람들이 나서서 하고 바이럴이 되기도 함.
5. 온보딩 과정을 컨텐츠화 하여 유저의 재미와 이탈 방지를 유도하며 Cold start도 해결하고자 함
![image](https://github.com/dhkim77000/winery/assets/89527573/38e3f7dc-746e-4f7a-8c02-52da3e9cea5a)

### 텍스트 임베딩
1. 와인은 매우 복합적인 향과 맛에 대한 내용은 단순한 수치형 데이터로는 표현이 안되는 부분이 존재.
2. 오히려, 와인에 대한 리뷰들에서 보다 많은 와인 정보들을 얻을 수가 있음.
3. BERT의 Multiclass-classification과 MLM(Masked Language Model)을 이용해 와인 리뷰 데이터들을 학습 및 임베딩하여 이를 feature로 활용

### 평점 불균형
![image](https://github.com/dhkim77000/winery/assets/89527573/66094a61-d77c-4e73-b8e5-401c1fb527c6)

## ML flow
![image](https://github.com/dhkim77000/winery/assets/89527573/d5f475e5-2544-4c34-9772-acfc8f863d42)

### Candidate Generation
1. 와인은 유투브와 다르게 기존의 소비에서 벗어난 너무 새로운 것이 추천되면 악영향이 발생할 수도 있음.
2. 이를 해결하기 위해 기존 소비에서 크게 벗어나지 않도록 후보군을 생성
3. 유저가 소비/좋아요한 아이템의 벡터들을 클러스터링을 통해서 군집 생성
4. 각 군집 별 유사 아이템을 벡터 간 연산을 통해 결과 추출
5. 마지막으로 유저 타입별로 Rule Based 필터링을 통해 최종 Candidate 생성. Rule은 EDA를 통해서 얻은 인사이트를 활용함(Ex. 초보 유저일수록 디저트, 스파클링 선호)
   
### Ranking
1. 다양한 메타데이터를 활용할 수 있고 정확성과 효율성 측면에서 목적에 적합했던 Context base 모델을 활용
2. DCN 모델을 기반으로 앞서 언급한 Text Embedding에서 추출한 벡터를 pre-trained 임베딩으로 활용
   
## Service Architecture
![image](https://github.com/dhkim77000/winery/assets/89527573/41d180f8-c1ad-4912-841f-f6b5718df87e)
1. 유저 로그는 MongoDB에 저장되고 batch 마다 GCP Bucket에 업로드 됨
2. 모델 또한 Airflow를 활용해 Batch 마다 GCP Bucket에 업로드된 데이터를 다운 받아 학습하고 새로 업데이트됨
3. 백엔드는 FastAPI를 이용해 구현, 프론트엔드는 ReactNative를 활용

## Contributors
- 김동환 | Crawling 및 데이터셋 구축, EDA, Text Embedding, 추천 시스템 모델링 및 배치화, 로그인/회원가입 API, 유저 로그 데이터베이스 구축
- 김영서 | 전체적인 데이터베이스 설계/구현, 추천 API, EDA
- 박재성 | DB 스키마 설계/구현, 전체적인 API, 모델 Inference 구현
- 전예원 | Frontend 설계, 모바일 UI 구현, EDA
- 진성호 | Frontend 설계, 개발 서버 생성/관리, EDA
