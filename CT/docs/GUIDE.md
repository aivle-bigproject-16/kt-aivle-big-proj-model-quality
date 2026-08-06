# 실행 가이드 — CT 품질 게이트를 처음 쓰는 사람에게

이 문서만 따라 하면 CT 사진 한 폴더를 넣어 "PASS / FAIL" 판정 CSV를 받을 수 있다.
설계 근거는 [PLAN.md](PLAN.md), 요약은 [README.md](README.md).

---

## 0. 이 도구가 하는 일

CT 사진을 넣으면 **"이 사진이 검사에 쓸 만큼 잘 찍혔나"** 만 판정한다.

```
CT 사진  →  [품질 게이트]  →  PASS  →  YOLO 결함탐지로 진행
                          →  FAIL  →  재촬영
```

**중요 — 이건 결함탐지가 아니다.** 배터리가 불량인지 정상인지는 전혀 보지 않는다.
오직 **촬영이 실패했는지**만 본다. 판정 대상인 촬영 실패는 5종이다.

| 실패 케이스 | 무엇이 잘못된 사진인가 |
|---|---|
| `ct_cell_alignment_failure` | FOV 정렬 실패로 배터리가 촬영 범위 밖에서 잘림 |
| `ct_acquisition_motion` | 촬영 중 움직여서 이중 영상이 겹침 |
| `ct_insufficient_projection_sampling` | 투영 수/각도 부족 → 줄무늬 + 미세구조 소실 |
| `ct_low_signal_noise` | 관전류·노출 부족 → 어둡고 노이즈 심함 |
| `ct_beam_hardening_metal_streak` | 금속 주변 beam hardening → 방사형 streak |

**출력은 확률이 아니라 판정선이 필요하다.** 게이트를 세게 잡으면 나쁜 사진을 잘 걸러내지만
멀쩡한 사진도 재촬영시켜 수율이 깎인다. 그 균형점을 정하는 게 아래 "운영점"이다.

---

## 1. 준비 — 이미 끝나 있음

| 항목 | 상태 |
|---|---|
| Python 가상환경 `.venv` | ✅ 생성됨 |
| PyTorch 2.13.0 (CPU판) | ✅ 설치됨 |
| 데이터 인덱스 `manifests/` | ✅ 20,000행 |
| 셀 단위 분할 `splits/` | ✅ 생성됨 |
| 전처리 캐시 `cache/CT_288x512.npy` | ✅ 8.8 GB |

**GPU는 없다.** 이 노트북(i5-1340P)의 CPU 12코어로 돌린다. 학습은 느리지만
분류(추론)는 빠르다 — 장당 수십 ms.

새 PC에서 처음부터 세팅한다면 [README.md](README.md)의 "환경" 절을 따른다.
한 가지만 주의: **`pip install --upgrade pip`을 venv 안에서 하지 말 것.**
Windows에서 pip이 자기 자신을 덮어쓰다가 깨진다(이미 한 번 겪었다).

---

## 2. 가장 자주 쓸 명령 — 사진 분류하기

PowerShell을 열고 프로젝트 폴더로 이동한 뒤:

```powershell
cd C:\quality_gate_mobilenet

.\.venv\Scripts\python.exe src\predict.py `
    --ckpt runs\CT_288x512_fold0\best.pt `
    --images "C:\내\CT사진폴더" `
    --out 결과.csv
```

끝이다. 세 가지만 지정한다.

| 옵션 | 뜻 |
|---|---|
| `--ckpt` | 학습된 모델 파일. 학습을 돌리면 `runs\...\best.pt`에 생긴다 |
| `--images` | 분류할 `.jpg`가 든 폴더 (파일 하나만 줘도 됨) |
| `--out` | 결과를 저장할 CSV 경로 |

> PowerShell에서 줄 끝의 백틱(`` ` ``)은 "다음 줄에 계속"이라는 뜻이다.
> 한 줄로 다 쓸 거면 백틱 없이 이어 쓰면 된다.

### 결과 CSV 읽는 법

```csv
file,logit,fail_prob,verdict
CT_cell_pouch_1900000001_x_2000019740.jpg,-4.2031,0.0148,PASS
CT_cell_pouch_1900000002_y_2000014273.jpg,2.8815,0.9469,FAIL
```

| 열 | 뜻 |
|---|---|
| `file` | 파일명 |
| `logit` | 모델 원점수. 클수록 FAIL 쪽. 판정은 이 값으로 한다 |
| `fail_prob` | 위를 0~1로 바꾼 값. **눈으로 보기 편하라고 넣은 참고값이다** |
| `verdict` | 최종 판정 `PASS` / `FAIL` |

실행하면 요약도 같이 찍힌다.

```
운영점: CT_288x512_fold0 val의 clean FPR 3% threshold = 0.8123 (logit)
총 1000장  ->  PASS 947 / FAIL 53 (5.3%)
결과 -> 결과.csv
```

### 판정선(운영점)을 바꾸고 싶다면

```powershell
.\.venv\Scripts\python.exe src\predict.py --ckpt ... --images ... --out ... --threshold 1.5
```

`--threshold`를 주지 않으면 **"멀쩡한 사진의 3%까지만 재촬영시키는 선"** 을 자동으로 쓴다.
이 3%가 어디서 나왔는지, 어떻게 바꾸는지는 §5에서 설명한다.

---

## 3. 모델을 직접 학습시키려면

이미 학습된 `best.pt`가 있으면 이 절은 건너뛰어도 된다.

```powershell
# fold 하나만 (약 8~9분/epoch × 8 epoch)
.\.venv\Scripts\python.exe -u src\train.py --size 288x512 --fold 0 --epochs 8
```

돌리면 이렇게 찍힌다. 아래는 2026-07-31 fold0 실제 출력이다.

```
[CT @ 288x512] fold0: train=fold2+fold3+fold4 val=fold1 test=fold0
  train 9969장/23셀  val 3059장/8셀  test 3172장/7셀  (lockbox 제외)  FAIL 비율 train 0.097
ep 1 loss 0.1678  481s  val PR-AUC 0.9840  recall@FPR3% 0.9690
```

epoch 하나에 약 8분이다(GPU 없이 CPU 12코어).

- `-u`는 "출력을 바로바로 보여줘라"는 뜻이다. 안 붙이면 한참 아무것도 안 보인다.
- **매 epoch 끝에 성적이 가장 좋은 상태가 `best.pt`로 저장된다.**
  그래서 8 epoch을 다 안 기다리고 중간에 `Ctrl+C`로 끊어도 그때까지의 최선이 남아 있다.

### 5-fold 전체

```powershell
.\.venv\Scripts\python.exe -u src\run_ct_cv.py --size 288x512 --epochs 8
```

fold 5개를 순서대로 돌리고 마지막에 **평균 ± 표준편차**를 찍는다.
전부 도는 데 몇 시간 걸리므로 시간 여유가 있을 때 돌린다.

**왜 5번이나 돌리나?** CT 사진 20,000장이 사실은 **배터리 47개**에서 나왔기 때문이다.
한 배터리에서 평균 425장의 인접 슬라이스가 나온다. 즉 실제로 독립적인 표본은 47개뿐이라,
어느 배터리가 시험지에 들어가느냐에 따라 점수가 크게 흔들린다.
그래서 5번 돌려 **평균과 흔들림 폭을 같이** 봐야 한다. 숫자 하나만 보면 속는다.

---

## 4. 처음부터 전부 다시 만들려면

데이터 폴더가 바뀌었거나 새 PC에서 시작할 때만 필요하다. 순서대로.

```powershell
# ① 인덱스 — 어떤 파일이 PASS/FAIL인지 목록 작성 (약 4분)
py -3.14 src\build_index.py

# ② 분할 — 배터리 단위로 학습/검증/시험 나누기 (즉시)
py -3.14 src\make_splits.py

# ③ 캐시 — 사진을 전부 같은 크기로 맞춰 한 덩어리로 저장 (약 30초, 8.8GB)
.\.venv\Scripts\python.exe -u src\cache_letterbox.py CT 288x512

# ④ 학습
.\.venv\Scripts\python.exe -u src\train.py --size 288x512 --fold 0 --epochs 8
```

①②는 `py -3.14`(시스템 파이썬), ③④는 `.\.venv\Scripts\python.exe`(가상환경)를 쓴다.
①②는 외부 라이브러리가 필요 없어서 그렇다. 헷갈리면 **전부 `.\.venv\Scripts\python.exe`로 써도 된다.**

각 단계가 하는 일:

**① 인덱스** — 사진 20,000장의 라벨을 모은다. 라벨은 세 군데(`labels_json`의 `quality_class`,
`pass`/`fail` 폴더, `augmentation_json` 존재 여부)에 있는데 **셋을 대조해서 어긋나면 알려준다.**
현재 데이터는 불일치 0건이다.

**② 분할** — 여기가 이 프로젝트에서 가장 중요한 단계다. §6에서 따로 설명한다.

**③ 캐시** — CT 사진은 높이가 전부 512인데 폭만 46~282로 제각각이다.
학습하려면 크기가 같아야 하므로 **288×512 캔버스 가운데에 놓고 남는 곳을 회색으로 채운다**(letterbox).
잘라내지 않는 게 핵심이다 — 잘라내면 "배터리가 화면 밖으로 나간" FAIL의 증거가 같이 잘려나간다.

**④ 학습** — MobileNetV3-Small(이미 ImageNet으로 사전학습된 모델)의 마지막 층만
PASS/FAIL 2지선다로 바꿔 미세조정한다.

---

## 5. 성적표 읽는 법 — 여기가 제일 중요하다

### 정확도(accuracy)는 절대 쓰지 않는다

데이터의 90%가 PASS다. **"전부 PASS"라고만 찍는 바보 모델도 정확도 90%가 나온다.**
그래서 정확도는 이 문제에서 아무 의미가 없다. 대신 두 숫자를 **따로** 본다.

| 지표 | 뜻 | 나빠지면 생기는 일 |
|---|---|---|
| **fail recall** | 실제 FAIL 중 몇 %를 잡았나 | 낮으면 → 나쁜 사진이 YOLO로 새어 나감 |
| **clean FPR** | 멀쩡한 사진 중 몇 %를 FAIL로 잘못 찍었나 | 높으면 → 멀쩡한 셀을 재촬영 = 수율 손실 |

이 둘은 **맞바꾸는 관계**다. 판정선을 낮추면 recall이 오르지만 FPR도 오른다.
그래서 **"FPR을 몇 %까지 감수할지 먼저 정하고, 그 안에서 recall을 최대로"** 가 올바른 순서다.

### 그래서 출력이 이렇게 생겼다

아래는 **형식을 보여주는 예시이며 숫자는 실제 결과가 아니다**
(학습이 끝나면 `runs\CT_288x512_fold0\metrics_test.json`에 진짜 값이 들어간다).

```
[TEST — held-out cells]
  n=3172 fail=321 PR-AUC=0.xxxx
  fail recall @ clean FPR 1% = 0.xxxx     ← 재촬영 1%까지 허용할 때 잡는 FAIL 비율
  fail recall @ clean FPR 3% = 0.xxxx     ← 3%까지 허용할 때
  fail recall @ clean FPR 5% = 0.xxxx     ← 5%까지 허용할 때
  케이스별 recall @ FPR 3%:
    ct_acquisition_motion               0.xxx (n=66)
    ct_beam_hardening_metal_streak      0.xxx (n=61)
    ct_cell_alignment_failure           0.xxx (n=63)
    ct_insufficient_projection_sampling 0.xxx (n=68)
    ct_low_signal_noise                 0.xxx (n=63)
```

**1%/3%/5% 세 줄을 다 찍는 이유**: 현장에서 "재촬영을 몇 %까지 감수할 수 있는가"가
아직 정해지지 않았기 때문이다. 정해지면 **그 줄만 보면 되고 재학습은 필요 없다.**
숫자는 `runs\CT_288x512_fold0\metrics_test.json`에도 저장된다.

**케이스별 recall이 핵심 산출물이다.** "전체 84%"보다 "어떤 촬영 실패를 놓치는가"가
현장에서 훨씬 쓸모 있다. 위 예시라면 정렬 실패를 절반밖에 못 잡으니 그게 개선 과제다.

---

## 6. 왜 데이터에 딸려온 test 폴더를 안 쓰는가

데이터에는 이미 `CT/test/` 폴더(1,000장)가 있다. **그런데 쓰면 안 된다.**

확인해 보니 test에 있는 배터리 47개가 **100% 전부 main(학습용)에도 들어 있다.**
같은 배터리의 다른 단면이 학습지와 시험지에 동시에 들어간 것이다.
이러면 모델이 "촬영 품질"을 배운 게 아니라 **그 배터리를 외운 것**인데도 점수가 잘 나온다.
증강계획서 v1.9 §1.1도 "`quality_class`만 90:10으로 맞추며 원본 split은 층화하지 않는다"고
명시하고 있어서, 의도된 동작이지 버그는 아니다. 다만 우리 목적에는 못 쓴다.

그래서 `make_splits.py`가 **배터리 단위로 다시 나눈다.**

```
fold0  7셀 3,172장   fold3  7셀 3,236장
fold1  8셀 3,059장   fold4  8셀 3,445장
fold2  8셀 3,288장   lockbox 9셀 3,800장  ← 봉인
```

같은 배터리가 두 곳에 들어가지 않는다(교집합 0을 스크립트가 검증한다).

### lockbox는 건드리지 말 것

9개 셀(3,800장)은 **한 번도 쓰지 않고 봉인해 둔 최종 확인용**이다.
여기를 보면서 모델을 고치기 시작하면, 이것도 결국 학습지가 되어 버린다.
**모든 개발이 끝난 뒤 딱 한 번만 개봉한다.**

---

## 7. 지금 나오는 숫자를 어디까지 믿을 것인가

**FAIL 2,000장은 전부 컴퓨터로 합성한 것이다. 실제로 잘못 찍힌 사진은 0장이다.**

증강계획서 v1.9 §2.2에 각 실패 케이스를 만드는 절차가 수식 수준으로 적혀 있는데,
그렇게 만들면 **그 프로그램 특유의 흔적(지문)** 이 남는다.
예를 들어 `ct_cell_alignment_failure`는 잘라낸 뒤 LANCZOS라는 방식으로 원래 크기에
되돌리는데, 여기서 특유의 링잉 패턴이 생긴다.

모델이 "정렬이 틀어졌다"를 배운 게 아니라 **"LANCZOS 흔적이 있다"** 를 배웠다면,
합성 데이터에서는 99%가 나오고 실제 현장에서는 무너진다.

그걸 재보려고 만든 게 이 명령이다.

```powershell
.\.venv\Scripts\python.exe -u src\ct_rule_baseline.py --size 288x512
```

세 가지를 나란히 학습시켜 비교한다.

| 실험 | 쓰는 정보 |
|---|---|
| 전체 피처 | 촬영 품질 지표 + 지문, 전부 |
| 지문 제외 | 촬영 품질 지표만 |
| **지문만** | 리샘플 흔적 2개만 |

**"지문만"이 "전체"에 가까운 점수를 내면, 그 성능은 촬영 품질이 아니라
증강 프로그램의 흔적을 잰 것이다.** 그때는 숫자를 신뢰하면 안 된다.

→ 결론: **실제로 잘못 찍힌 CT 사진을 소량이라도 확보하기 전까지,
여기 나오는 숫자를 실배포 성능의 근거로 쓰지 않는다.**

---

## 8. 자주 막히는 곳

| 증상 | 원인 / 해결 |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | `python` 대신 `.\.venv\Scripts\python.exe`로 실행할 것 |
| 실행했는데 아무것도 안 나온다 | `-u` 옵션을 붙일 것. 출력이 버퍼에 갇혀 있다 |
| `threshold를 못 찾음` | 학습이 끝나야 `metrics_val.json`이 생긴다. 아니면 `--threshold` 직접 지정 |
| `FileNotFoundError: cache\CT_288x512.npy` | §4의 ③ 캐시 단계를 먼저 실행 |
| 학습이 너무 느리다 | 정상이다(GPU 없음, 8~9분/epoch). `--epochs`를 줄이거나 중간에 `Ctrl+C` — `best.pt`는 이미 저장돼 있다 |
| 디스크가 부족하다 | `cache\` 폴더가 8.8GB다. 지워도 §4 ③으로 30초면 다시 만든다 |
| **로그의 한글이 `?쒖쇅`처럼 깨진다** | Windows PowerShell 5.1은 `Get-Content` 기본 인코딩이 CP949다. **`-Encoding UTF8`을 붙일 것.** 또는 `.\watch.ps1` 사용, 또는 PowerShell 7(`pwsh`)로 실행 |
| 학습 진행이 안 보인다 | epoch 하나가 8분이라 조용하다. `.\watch.ps1`로 실시간 확인 |

---

## 9. 폴더 구조

```
C:\quality_gate_mobilenet\
├─ .venv\              파이썬 가상환경 (torch 등)
├─ src\                코드
├─ manifests\          사진 목록 + 라벨 (index_ct.csv)
├─ splits\             배터리 단위 분할표 (ct_folds.csv)
├─ cache\              전처리된 사진 덩어리 (8.8GB, 지워도 재생성 가능)
├─ runs\               학습 결과 — best.pt, metrics_*.json
├─ GUIDE.md            ← 지금 이 문서
├─ README.md           요약 + CT 설계 결정 근거
└─ PLAN.md             전체 설계·리스크 분석
```

원본 데이터는 `C:\quality_fail_40k_v1.8_20260730_unzip\` 에 있고 **읽기만 한다.**
