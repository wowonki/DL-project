# 개인 특성으로 신용 대출 수요 예측하기
작성자: 이원기, 경제금융학부, dnjs2658@naver.com

## Proposal

- 최근 개인 신용대출이 많이 늘었습니다. 코로나 사태로 인해서 개개인의 생활고가 심해진 탓도 분명 있겠지만,  
불어오는 투자 열풍에 소위 '빚투' 라고 불리는 '빚을 지어 투자' 하는 행태가 늘어난 영향도 무시하지 못 할 것입니다.  
대출을 받을 때 여신자와 수신자는 서로간의 정보의 불균형 상태에 빠지게 됩니다.  
수신자가 알 수 있는 것은 금리와 상환일 등 피상적인 정보 뿐이고,  
여신자가 알 수 있는 것도 개개인이 입력한 개인정보 정도 밖에 없습니다.  
이 상황에서 수신자는 자신의 조건과 꼭 맞는 대출상품을 찾는데 어려움이 있고,  
여신자 입장에서도 좋은 대출상품을 맞는 고객을 찾는데에 어려움을 느낍니다.  
그래서 이번 프로젝트를 통해 고객 개개인의 특성을 알 때 최적의 대출상품을 추천하기 위해  
기본이 되는 신용대출 수요를 파악하는 모델을 만들고,  
이를 발전시켜 범기업적 측면에서 개개인에게 상품을 추천해주는 모델까지 개발하고 싶습니다.

## Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 데이터 불러오기
```python
df=pd.read_csv('../loan_data.csv')
df.head()
```

데이터의 구체적인 내용은 다음과 같다
```
feature:
  id: 자료 인덱스 넘버
  age: 고객의 나이
  experience: 신용대출 횟수
  income: 고객의 연 수입
  zip_code:
  family: 가족 구성원 수(본인 포함)
  cc_avg:
  education; 1: high school graduate, 2: college graduate, 3: university graduate
  martage: 현재 저당잡힌 액수
  securities_account: 증권계좌보유여부
  cd_account: CD계좌보유여부
  credit_card: 신용카드보유여부
  
target:
  personal_loan: 대출수요
```

### 결측치 확인
```python
df.info()
```

결측치가 없는 것을 확인할 수 있다.

### 시각화
```
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), vmin=-1, cmap="plasma_r", annot=True)
```
이를 보면 age와 experience feature간의 상관관계가 과도하게 높은 것을 알 수 있다.
저정도로 높은 선형관계를 가지는 데이터가 있을 경우 분석에 방해가 될 수 있기에
age feature를 제거하도록 한다.
```
df.drop(['age'], axis=1, inplace=True)
````

```
plt.figure(figsize=(8,5))
sns.countplot('securities_account', data = df, color='#00ddff', saturation=0.9)
```

```
sns.scatterplot(x = 'personal_loan', y = 'inocme', data = df)

print(df[df['personal_loan']==0]['income'].mean())
print(df[df['personal_loan']==1]['income'].mean())
```
대출을 받는 사람들의 평균 소득이 그렇지 않은 사람들보다 높다는 유의미한 결과를 얻을 수 있다.
