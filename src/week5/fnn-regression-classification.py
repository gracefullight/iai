# 미래 버전의 파이썬 기능을 사용할 수 있게 해줍니다 (타입 힌트 관련)
from __future__ import annotations

# 파일 경로를 다루기 위한 pathlib 모듈
from pathlib import Path

# 데이터 처리, 모델 생성, 평가에 필요한 라이브러리들
import pandas as pd  # 데이터프레임 처리
import tensorflow as tf  # 딥러닝 프레임워크
from keras.layers import Dense  # 신경망의 Dense(완전연결) 레이어
from sklearn import metrics  # 평가 지표
from sklearn.model_selection import train_test_split  # 데이터 분할
from tensorflow.keras.models import load_model  # 저장된 모델 불러오기

# 문제 유형 상수 정의
PROBLEM_REGRESSION_SINGLE = 1
PROBLEM_REGRESSION_MULTI = 2
PROBLEM_CLASSIFICATION = 3

# 데이터 파일들이 저장된 assets 폴더 경로 지정
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
# 예시로 airline-passengers.csv 파일에서 두 번째 컬럼만 읽어옴
dataframe: pd.DataFrame = pd.read_csv(
    ASSETS_DIR / "airline-passengers.csv", usecols=[1], engine="python"
)

# 어떤 문제를 풀 것인지 사용자에게 입력받음
problem_code = int(
    input(
        "Input a problem code: problem_code = 1 -- a regression problem with single output, "
        "problem_code = 2 -- a regression problem with multi-outputs, "
        "problem_code = 3 -- a multi-class classification problem. Your choice: "
    )
)

# 문제 유형에 따라 데이터셋을 다르게 불러옴
if problem_code == PROBLEM_REGRESSION_SINGLE:
    # 1번: 단일 출력 회귀 문제. HousingData.csv 파일을 불러옴
    df = pd.read_csv(ASSETS_DIR / "HousingData.csv", names=["x1", "x2", "y"])
    print(df.shape)  # 데이터프레임의 크기 출력
    df.head()  # 데이터프레임의 앞부분 출력
    # 입력(X): 첫 두 컬럼, 출력(y): 마지막 컬럼
    X = df.iloc[:, 0:2]
    y = df.iloc[:, -1]
elif problem_code == PROBLEM_REGRESSION_MULTI:
    # 2번: 다중 출력 회귀 문제. 사이킷런의 linnerud 데이터셋 사용
    from sklearn.datasets import load_linnerud  # 운동 데이터셋 불러오기

    # load_linnerud 함수로 입력(X), 출력(y) 데이터를 프레임 형태로 반환
    X, y = load_linnerud(return_X_y=True, as_frame=True)
elif problem_code == PROBLEM_CLASSIFICATION:
    # 3번: 다중 클래스 분류 문제. wine.csv 파일을 불러옴
    df = pd.read_csv(
        ASSETS_DIR / "wine.csv",
        names=[
            "y",
            "x1",
            "x2",
            "x3",
            "x4",
            "x5",
            "x6",
            "x7",
            "x8",
            "x9",
            "x10",
            "x11",
            "x12",
            "x13",
        ],
    )
    print(df.shape)  # 데이터프레임의 크기 출력
    df.head()  # 데이터프레임의 앞부분 출력
    X = df.iloc[:, 1:]  # 입력(X): 첫 번째 컬럼 제외
    y = df.iloc[:, 0]  # 출력(y): 첫 번째 컬럼
    y = y - 1  # 클래스 라벨을 0부터 시작하도록 조정
else:
    print("Invalid problem code. Try a correct one: 1 or 2")

# 데이터를 학습용/테스트용으로 8:2로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 테스트셋에서 마지막 2개 샘플을 미래 예측용으로 따로 저장
future_sample_data = X_test[-2:]
future_sample_label = y_test[-2:]
print(future_sample_data)  # 미래 예측용 데이터 출력
print(future_sample_label)  # 미래 예측용 정답 출력
# 나머지 테스트셋으로 평가 진행
X_test = X_test[:-2]
y_test = y_test[:-2]

# 문제 유형에 따라 신경망 구조와 하이퍼파라미터 설정
if problem_code == PROBLEM_REGRESSION_SINGLE:
    input_dim = X_train.shape[1]  # 입력 차원
    n_neurons_1 = 512  # 첫 번째 은닉층 뉴런 수
    n_neurons_2 = 512  # 두 번째 은닉층 뉴런 수
    n_neurons_3 = 100  # 세 번째 은닉층 뉴런 수
    output_dim = 1  # 출력 차원(회귀)
    epochs = 20  # 학습 반복 횟수
    batch_size = 10  # 배치 크기
elif problem_code == PROBLEM_REGRESSION_MULTI:
    input_dim = X_train.shape[1]
    n_neurons_1 = 512
    n_neurons_2 = 512
    n_neurons_3 = 100
    output_dim = 3  # 다중 출력 회귀
    epochs = 20
    batch_size = 10
elif problem_code == PROBLEM_CLASSIFICATION:
    input_dim = X_train.shape[1]
    n_neurons_1 = 512
    n_neurons_2 = 512
    n_neurons_3 = 100
    output_dim = 3  # 다중 클래스 분류
    epochs = 20
    batch_size = 16
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")

# 신경망 모델 생성 (순차적으로 레이어 쌓기)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(input_dim,)))  # 입력 레이어
model.add(Dense(n_neurons_1, activation="relu"))  # 첫 번째 은닉층
model.add(Dense(n_neurons_2, activation="relu"))  # 두 번째 은닉층
model.add(Dense(n_neurons_3, activation="relu"))  # 세 번째 은닉층
if problem_code in {PROBLEM_REGRESSION_SINGLE, PROBLEM_REGRESSION_MULTI}:
    model.add(Dense(output_dim))  # 회귀: 출력층(활성화 함수 없음)
elif problem_code == PROBLEM_CLASSIFICATION:
    model.add(Dense(output_dim, activation=tf.keras.activations.softmax))  # 분류: 출력층(softmax)
else:
    print("invalide code")

# 모델에 포함된 레이어 수 출력
print(f"The model layers is {len(model.layers)}: ")

# 모델 컴파일 (손실 함수, 최적화 방법, 평가 지표 설정)
if problem_code in {PROBLEM_REGRESSION_SINGLE, PROBLEM_REGRESSION_MULTI}:
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error"])
elif problem_code == PROBLEM_CLASSIFICATION:
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")
model.summary()  # 모델 구조 요약 출력

# 모델 학습 (fit 함수)
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

# 테스트셋에 대해 예측 수행
if problem_code in {PROBLEM_REGRESSION_SINGLE, PROBLEM_REGRESSION_MULTI}:
    pred = model.predict(X_test)  # 회귀: 예측값
elif problem_code == PROBLEM_CLASSIFICATION:
    predictions = model.predict(X_test)  # 분류: 각 클래스별 확률
    y_preds = predictions.argmax(axis=1)  # 확률이 가장 높은 클래스 선택
    scores = model.evaluate(X_test, y_test, verbose=0)  # 평가 지표 출력
    print(scores)
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")

# 예측 결과 평가 및 시각화
if problem_code in {PROBLEM_REGRESSION_SINGLE, PROBLEM_REGRESSION_MULTI}:
    r_square_test = metrics.r2_score(y_test, pred)  # R2 점수
    print(f"r_square_test is {r_square_test}:")
    mean_absolute_error_test = metrics.mean_absolute_error(y_test, pred)  # MAE
    print(f"mean_absolute_error_test is {mean_absolute_error_test}:")
    mean_squared_error_test = metrics.mean_squared_error(y_test, pred)  # MSE
    print(f"mean_squared_error_test is {mean_squared_error_test}:")
    import matplotlib.pyplot as plt  # 산점도 시각화

    plt.scatter(y_test, pred)
    plt.xlabel("y_test")
    plt.ylabel("pred")
    plt.show()
elif problem_code == PROBLEM_CLASSIFICATION:
    accuracy_test = metrics.accuracy_score(y_preds, y_test)  # 정확도
    print(f"accuracy_test is {accuracy_test}:")
    precision_test = metrics.precision_score(y_test, y_preds, average="weighted")  # 정밀도
    print(f"precision_test is {precision_test}:")
    recall_test = metrics.recall_score(y_test, y_preds, average="weighted")  # 재현율
    print(f"recall_test is {recall_test}:")
    f1_score_test = metrics.f1_score(y_test, y_preds, average="weighted")  # F1 점수
    print(f"f1_score_test is {f1_score_test}:")
    import matplotlib.pyplot as plt  # 혼동행렬 시각화
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    cm = confusion_matrix(y_test, y_preds)
    cmp = ConfusionMatrixDisplay(cm, display_labels=["class_0", "class_1", "class_2"])
    cmp.plot(ax=ax)
    evaluation_report = metrics.classification_report(y_test, y_preds)  # 평가 리포트
    print(evaluation_report)
else:
    print("Invalid problem code. Try a correct one: 1 or 2 or 3")

# 학습된 모델 저장
model.save(ASSETS_DIR / "model.keras")

# 저장된 모델 불러오기 및 구조 출력
loaded_model = load_model(ASSETS_DIR / "model.keras")
loaded_model.summary()

# 미래 샘플에 대해 저장된 모델로 예측 수행
if problem_code in {PROBLEM_REGRESSION_SINGLE, PROBLEM_REGRESSION_MULTI}:
    preds_future = loaded_model.predict(future_sample_data)
elif problem_code == PROBLEM_CLASSIFICATION:
    predictions_val = loaded_model.predict(future_sample_data)
    preds_future = predictions_val.argmax(axis=1)
    print(
        f"The predicated classes are {preds_future} vs the true classes are "
        f"{future_sample_label.to_numpy()}"
    )
else:
    print("Invalid code")

# 미래 샘플별로 예측값과 실제값 출력
for i in range(2):
    print(
        f"The future data is {future_sample_data.to_numpy()[i]}, "
        f"the predicted value is {preds_future[i]} and "
        f"the actual value is {future_sample_label.to_numpy()[i]}"
    )
