# 개인 특성으로 대출 상환 예측하기
작성자: 이원기, 경제금융학부, dnjs2658@naver.com

## 목적

2022년 현재까지 코로나 사태로 인해서 빚어진 개개인의 생활고와  
투자 열풍으로 인한 '빚투' 라고 불리는 '빚을 지어 투자' 하는 행태로 인해  
개인 신용대출 규모가 상당히 커졌다.   

대출을 받은 수신자는 신용을 바탕으로 대출금을 받는 대신  
만기가 되면 여신자에게 대출금을 상환해야 하는 의무를 진다.  
보통의 경우에는 만기에 맞게 대출금을 상환하지만, 아닌 경우도 존재한다.  
만기가 도래했을 때 갚을 대출금이 부족하거나, 심한 경우 파산을 신청해  
채무를 변제받는 경우도 있다.  
이 경우에 신용을 바탕으로 금전을 빌려준 여신자가 부담을 떠안게 된다.  

때문에 정보 불균형으로 인해 여신자에게 발생하는 비용을 최소화시키기 위해서  
고객 개개인의 특성을 알 때 여신자가 수신자의 채무 불이행 여부를 예측해 보고자 한다.


## 데이터 셋

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
credit.policy: LendingClub.com에 의거한 신용평가기준 만족여부(1: 만족, 0: 불만족)
purpose: 대출의 목적
int.rate: 대출 금리
installment: 수신자가 지불해야 하는 월별 할부금
log.annual.inc: 수신자 연봉의 로그값
dti: 부채-수입 비율
fico: FICO에 의거한 신용점수
days.with.cr.line: 수신자에게 신용 한도액이 있었던 일 수
revol.bal: 수신자의 회전 잔액
revol.util: 수신자의 회전 회선 이용률
inq.last.6mths: 최근 6개월 동안 여신자의 대출 조회 수
delinq.2yrs : 지난 2년 동안 수신자가 지불 기한을 30일 이상 넘긴 횟수
pub.rec: 수신자의 부정적인 공적 기록 수 (탈세 기록, 전과 등)

not.fully.paid: 수신자의 채무 불이행 여부 (1: 불이행, 0:이행)
```

### 결측치 확인
```python
df.info()
```
![df_info](https://user-images.githubusercontent.com/62041260/174081687-e057ef51-d57a-4035-899f-e0168f5c7167.png)  
결측치가 없는 것을 확인할 수 있다.

### 데이터 시각화
```python
# 데이터 분석을 위한 분류
categorical_columns = ['credit.policy','purpose', 'inq.last.6mths', 'delinq.2yrs', 'pub.rec']
numerical_columns = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util']
```

#### Categorical Data

```python
# categorical한 데이터들을 for문을 통해 subplot으로 나타내어 한 번에 확인
fig,axes = plt.subplots(3,2,figsize=(15,15))
for idx,cat_col in enumerate(categorical_columns):
    row,col = idx//2,idx%2
    sns.countplot(x=cat_col,data=df,hue='not.fully.paid',ax=axes[row,col])
```
![first_plot](https://user-images.githubusercontent.com/62041260/174081012-8e252e77-693d-41d4-830b-c66ce827f613.png)  
![first_plot2](https://user-images.githubusercontent.com/62041260/174082903-d5650f55-2f85-49c8-9fc2-c8c03d96f565.png)

- credit policy가 0일때 채무 불이행 비율이 상당히 높다  
- purpose 에 따라 상환 비율이 크게 바뀌므로 확인해볼 필요가 있다.  
- inq.last.6mth가 커질 수록 채무 불이행 비율이 높아진다. / 구간을 나눠 살펴볼 필요가 있다.  
- delinq.2ys는 큰 상관관계를 보이진 않는다.  
- pub.rec도 큰 상관관계를 보이진 않는다.

#### Numerical Data

```python
fig,axes = plt.subplots(4,2,figsize=(17,20))
for idx,cat_col in enumerate(numerical_columns):
    row,col = idx//2,idx%2
    sns.boxplot(y=cat_col,data=df,x='not.fully.paid',ax=axes[row, col])
```
![second_plot](https://user-images.githubusercontent.com/62041260/174081018-fa5d09f8-bf1f-4be0-b7a7-cb785969ed30.png)  
![second_plot2](https://user-images.githubusercontent.com/62041260/174081019-bd93b641-0b4a-4670-b04f-83773c31f29d.png)  

- int.rate가 높아질 수록 채무 불이행 비율이 높아진다.
- fico가 낮아질 수록 채무불이행 비율이 높아진다.
- 그 외에는 크게 유의한 점 없으나 re.vol.bal 그래프의 가시성이 떨어져 다른 plot을 사용해본다.

```python
sns.ecdfplot(x='revol.bal',data=df,hue='not.fully.paid')
```
![cum_plot](https://user-images.githubusercontent.com/62041260/174081024-f38ebf5e-d170-449d-b0eb-97202bfbc14a.png)  

revol.bal의 누적 분포를 그려보았다.  
채무 불이행 여부에 큰 영향이 없어 보인다.


int.rate와 채무불이행 여부의 관계를 다른 plot으로 살펴본다.
```
sns.histplot(x='int.rate',data=df,hue='not.fully.paid')
```
![int_hist](https://user-images.githubusercontent.com/62041260/174081031-a2feb951-c80e-4b06-a49c-649b59fec154.png)  

boxplot에서 파악한 내용을 histplot으로 구체적으로 확인해보았다.
int.rate가 커질 수록 연체자 비율이 높아진다는 사실을 구체화 했다.


```python
sns.heatmap(df.corr())
```
![heatmap](https://user-images.githubusercontent.com/62041260/174081025-beee86ab-08be-42bd-bfae-a448f5daa8ea.png)  

not.fully.paid와 연관 있어 보이는 것은 creditpolicy, fico 정도,
두 요소를 이용해 새로운 변인을 만들어 본다.  

```python
fico_dummy = df['fico'] > 700
df['credit_score'] = df['credit.policy'] + fico_dummy
df['credit_score']
sns.countplot(x='credit_score',data=df,hue='not.fully.paid')
```

![credit_score](https://user-images.githubusercontent.com/62041260/174081028-e95a4de4-4979-4657-9185-45388fff57a9.png)    

LendingClub.com 에서 신용평가기준을 만족하고 fico 점수가 700점이 넘는 사람을 고신용자로,  
둘 중에 한 조건만 만족시키는 사람을 중신용자로,  
모두 만족하지 못하는 사람을 저신용자로 분류해 0,1,2로 나타내보았다.  
신용도가 낮을 수록 채무불이행할 확률이 높다는 사실을 다시 한번 확인하였다.  


이번에는 inq.last.6mths와 delinq.2yrs 두 요소를 이용해 새로운 변인을 만들어본다.  
inq.last.6mths는 여신자의 대출조회 수를, delinq.2yrs는 채무자의 연체 횟수를 나타낸다.  
따라서 여신자의 대출 조회를 4회 이상 받거나, 연체를 1번이라도 했던 사람을 위험군으로 분류해본다.  
```python
# 4번 이상 받은 사람을 위험군으로 본다.
inq_6mth = df['inq.last.6mths'] >= 4

# 1번이라도 받았던 사람을 위험군으로 본다
del_2yr = df['delinq.2yrs'] != 0

# 6개월 채무 이행 콜을 4번 이상 받았거나, 2년 내에 연채를 한 사람
df['hazard_score'] = inq_6mth | del_2yr
df['hazard_score'] = df['hazard_score']*1
sns.countplot(x='hazard_score',data=df,hue='not.fully.paid')
```
![hazard_score](https://user-images.githubusercontent.com/62041260/174081036-eebddab0-992b-4d79-b6ca-8f5480ca4872.png)  

마지막으로 int.rate와 installment 요소들을 활용해  
높은 이자율로 대출을 받은 사람이 많은 월 할부금을 지급해야 할 때  
채무 불이행 비율이 높아지는지 알아본다.  

```python
sns.lmplot('installment','int.rate',data=df,hue='not.fully.paid',palette='coolwarm')
```
![int_install](https://user-images.githubusercontent.com/62041260/174081039-153c5236-9988-4768-93c7-fd582c5017a5.png)  

채무불이행 회귀선이 채무 이행 회귀선보다 조금 높게 위치하긴 하지만 크게 유의미하진 않은 듯 하다.  

여기까지 시각화를 바탕으로 분석한 결과, 개인 채무 불이행에 영향을 미치는 주 feature는  
credit policy, fico, int.rate, credit_score, hazard_score 정도이다.  
***
## 의사결정 트리로 예측
의사결정 트리란 특정한 문제를 해결하기 위해 한 단계씩 내려가면서 0또는 1의 의사결정을 한다.  
트리는 단계를 내려가며 점점 더 많은 노드들을 만들어 내게 되고, 이 중에서 가장 설명력이 높은  
노드를 선택하는 것이 의사결정 트리의 골자이다.  
이 알고리즘은 분류와 회귀 목적 모두에 사용될 수 있는 지도 학습 알고리즘이며,  
여기서는 신용대출 상환여부 분류를 위해 사용되었다.  


desicion tree 분석을 위해 유의해 보이는 데이터 셋을 추출해 새로운 데이터프레임을 만든다
hazard_score는 구성요소인 dummy columns을 따로 사용한다
```python
df['inq_6mth_4'] = (df['inq.last.6mths'] >= 4)*1
df['del_2yr_once'] = (df['delinq.2yrs'] != 0)*1

valid_col = ['credit.policy', 'int.rate', 'fico', 'credit_score', 'inq_6mth_4', 'del_2yr_once', 'not.fully.paid']
df2 = df[valid_col]
df2[['int.rate','fico']].describe()
```
![int_fico](https://user-images.githubusercontent.com/62041260/174081049-bac69626-000a-47ff-a99b-b24f287f914d.png)  

두 연속적인 feature를 25, 50, 75% 구간으로 4분할 해 더미로 만든다.
```python
int_qlist = [0.1039, 0.1221, 0.1407]
fic_qlist = [682, 707, 737]
df2['int_dummy'] = 0
df2['fico_dummy'] = 0
for i in int_qlist:   
    cond = df2['int.rate'] >= i
    df2.loc[cond, 'int_dummy'] += 1
    
for i in fic_qlist:   
    cond = df2['fico'] >= i
    df2.loc[cond, 'fico_dummy'] += 1

df2.drop(['int.rate', 'fico'], axis=1, inplace=True)
```

모델을 적용하기 이전에, 데이터를 테스트셋과 트레인셋으로 분류한다
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X=df2.drop('not.fully.paid', axis=1)
y=df2['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
```

지니계수를 기준으로 하는 의사결정모델을 만들고 트레인 셋, 테스트 셋을 적용해 본다.
```python
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

y_pred_gini = clf_gini.predict(X_test)
y_pred_train_gini = clf_gini.predict(X_train)
```

정확도를 출력해본다.
```python
from sklearn.metrics import accuracy_score

print('지니계수를 이용한 테스트셋 정확도: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))
print('트레이닝셋 정확도: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))
```
![predict_acc](https://user-images.githubusercontent.com/62041260/174087600-8d529222-58f9-423e-9648-4481963eda35.png)  

train 셋과 test 셋으로 예측한 결과가 overfit 하지 않고 적절한 수준을 보여준다

그림을 그려 의사결정트리를 살펴본다.  
```python
from sklearn import tree

plt.figure(figsize=(12,8))
tree.plot_tree(clf_gini.fit(X_train, y_train)) 
```  
![deci_tree](https://user-images.githubusercontent.com/62041260/174080999-7c37367b-6251-44a7-b562-aec0689f406a.png)  

***  
## 결론  

지금까지 개인 특성을 이용하여 신용대출 상환 여부를 판단하기 위해 데이터 전처리부터  
시각화를 통해 feature들의 특성을 파악하고, 새로운 feature를 만들어 분석해보았다.   
상환여부에 영향을 미치는 요소들은 대출금리, 채무불이행 건수 등이 있었지만,  
그 중 제일 유의했던 feature들은 credit.policy와 pico로 대표되는 신용점수였다.  
신용 점수가 낮으면 낮을 수록 채무 불이행 확률이 높아지는, 아주 직관적인 결과다.  

또한 decision tree를 이용해 각 feature들로 예측하는 모델을 적용해보았다.  
정확도는 약 70~80%를 보여주며 overfitting 하지 않은 유의한 결과를 만들어 냈다.  

한 학기 동안 수업을 들으면서 딥러닝에 관한 교양적 지식도 쌓고  
코딩을 하는 법도 조금은 배웠다고 생각한다.  
물론 코딩을 전문적으로 하는 사람들이 본다면 코웃음 칠 정도지만,  
적어도 내가 하고자 하는 프로젝트에 이용하는 도구로써는 어느 정도 사용할 수 있지 않을까.  
아무튼 의미있는 한 학기 수업이었다. 
