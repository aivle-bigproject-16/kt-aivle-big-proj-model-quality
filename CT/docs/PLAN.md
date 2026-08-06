# 품질 게이트 분류기 (MobileNetV3-Small) — 단계별 실행 계획

작성: 2026-07-30 / 갱신: 2026-07-31 (v1.9 반영)
근거 문서: `품질분류기_MobileNet_설계(0724)`, `데이터_증강_계획서_v1.9(2026-07-31)`
데이터: `C:\quality_fail_40k_v1.8_20260730_unzip` (1.2 GB, 실측 완료)

> **v1.9의 위치**: v1.9는 스스로 "변경 범위는 `run_pipeline.ps1` 실행 래퍼와 업로드 절차이며,
> 수량·증강 알고리즘·품질 게이트·좌표 변환·출력 JSON schema·ID 발급 규칙은 v1.8과 동일"이라고 명시한다.
> 즉 **v1.9 §2는 우리 v1.8 데이터를 만든 알고리즘의 정본 사양**이다.
> 공개 산출물이 `QF18_` ID와 `v1.8:<failure_case>` reference를 유지하는 것도 같은 이유다.
> → 학습 계획에 미치는 영향: 파이프라인 절차(§4.10~4.13)는 무관. **§2 증강 사양이 전부 유효 정보.**

---

## 0. 실측 결과 요약 (계획의 전제)

### 0.1 데이터 실측

| 항목 | CT | RGB |
|---|---|---|
| 총 이미지 | 20,000 | 20,000 |
| main / test | 19,000 / 1,000 | 19,000 / 1,000 |
| PASS / FAIL | 18,000 / 2,000 | 18,000 / 2,000 |
| **고유 battery_id** | **47** | **757** |
| 배터리당 이미지 | 52~575 (평균 425) | 4~45 (평균 26) |
| form | pouch 100% | cylindrical 100% |
| axis | x 2,450 / y 9,256 / z 8,294 | — |
| 해상도 | 가변 (72×512, 147×512 …) | 512×288 고정 |
| 실패 케이스 | 5종 | 8종 |

폴더 구조: `full_images/{CT,RGB}/{main,test}/{images,labels_json,augmentation_json,pass,fail}`
`pass/`·`fail/`는 `fail_image_split.ipynb`가 `images/`를 **복사**해 만든 중복본. 라벨의 정본은 `labels_json/*.json`의 최상위 `quality_class`.

### 0.2 실패 케이스 실제 분포

CT main(1,900): `acquisition_motion` 373 / `beam_hardening_metal_streak` 385 / `cell_alignment_failure` 380 / `insufficient_projection_sampling` 387 / `low_signal_noise` 375
CT test(100): 27 / 15 / 20 / 13 / 25

RGB main(1,900): `focus_failure` 299 / `hair_contamination` 185 / `overexposure` 254 / `reflection_glare` 251 / `surface_dust` 180 / `trigger_timing_failure` 217 / `underexposure` 255 / `uneven_lighting` 259
RGB test(100): 11 / 5 / 16 / 19 / 10 / 13 / 15 / 11

### 0.3 실행 환경

- Python 3.14 (`py -3.14`), 설치 패키지는 ipykernel 계열뿐 — **numpy/torch/PIL 없음**
- torch 2.13.0 / torchvision 0.28.0 cp314 휠 존재 (설치 가능 확인)
- CPU: i5-1340P (12C/16T), RAM 31.7 GB, C: 여유 66 GB
- **GPU: Intel Iris Xe 내장뿐 — CUDA 없음. CPU 학습 전제.**
  단 이미지가 이미 장변 512로 축소된 소용량(장당 5~15 KB)이라 CPU로도 실용적.

---

## 1. 먼저 정리해야 할 불일치 4건

### 1.1 🔴 test split의 battery_id 100% 누수 — **제공된 test는 평가에 쓸 수 없음**

실측:

| 모달 | main 고유 bid | test 고유 bid | test bid 중 main에도 있는 것 |
|---|---|---|---|
| CT | 47 | 47 | **47 (100%)** |
| RGB | 757 | 548 | **548 (100%)** |

증강계획서 §1.1이 명시한 대로("`quality_class`만 90:10으로 맞춘다")
셀 단위 층화를 하지 않았고, 그 결과 **test의 모든 배터리가 train에도 존재**한다.
CT는 배터리당 평균 425장의 인접 슬라이스라 사실상 같은 셀의 다른 단면을 외우면 맞는 구조.
→ **제공 test 무시하고 battery_id 기준으로 자체 재분할.** (설계문서 §3.3이 예측한 그대로)

### 1.2 🔴 CT의 유효 표본 수는 20,000이 아니라 **47**

CT 20,000장은 47개 셀에서 나왔다. 독립 단위는 47개.
→ 단일 hold-out split은 분산이 너무 크다. **GroupKFold(5) 교차검증으로 평균±표준편차 보고**가 필수.
→ RGB(757 셀)는 단일 3-way split으로 충분.

### 1.3 🟡 설계문서(0724)의 케이스 목록이 구버전(v1.5) 기준

| | 설계문서 0724 | 실제 데이터 v1.8 (= v1.9 §2) |
|---|---|---|
| CT | 8종 (positioning, fixation_motion, low_xray, low_resolution, detector_calibration, high_xray, sparse_projection, photon_starvation) | **5종** |
| RGB | 9종 (`rgb_alignment` 포함) | **8종** (`rgb_alignment` 없음) |

영향: 설계문서가 "기하 FAIL"로 꼽은 것 중 RGB는 `trigger_timing_failure` **하나만** 남았다(218장/main).
→ letterbox 필수 논거는 유지되나, 기하 케이스 recall은 표본이 작아 신뢰구간이 넓다는 점을 감안.

v1.9 §2를 읽고 나서 추가로 확인된 **케이스 정의 자체의 변경**(설계문서 0724 및 v1.7 대비):

| 케이스 | 변경 | 모델링 영향 |
|---|---|---|
| `ct_cell_alignment_failure` | v1.7의 "porosity를 FOV 밖으로 평행이동"에서 → **porosity 무관 edge crop 후 LANCZOS로 원본 raster 크기 복원**. padding 없음, 종횡비 유지 | 순수 기하가 아니라 **crop + 업스케일 보간 흐림**의 복합 신호. 종횡비가 보존되므로 종횡비 shortcut은 없음(다행). 대신 LANCZOS 지문 위험 |
| `rgb_focus_failure` | v1.7의 "25% 확률 motion blur 혼합" 삭제 → **`defocus_blur` 단일 단계** | 순수 등방 Gaussian blur만 남음. **라플라시안 분산 규칙으로 거의 결정적으로 잡힘** |
| `rgb_reflection_glare` | 전면 재설계. 선형 반사 띠(길이:폭 ≥5:1), core alpha `0.55~0.78`, **색 고정 `(255,246,224)`** | 아래 §2 지문 항목 참조 |
| `rgb_underexposure` | exposure factor `0.30~0.55` → `0.18~0.40` + **outline 평균비를 `0.44~0.56` 목표로 최대 3회 보정** | 밝기 통계가 좁은 구간으로 강제됨 → 규칙으로 매우 쉬움 |
| `rgb_overexposure` | **outline 내부 목표 포화율을 픽셀 개수로 정확히 맞춰 255로 만들고, 나머지 outline 픽셀은 249 이하로 clamp** | 아래 §2 지문 항목 참조 |
| `rgb_uneven_lighting` | 4방향 순환 탐색 + **(dark, bright) gain 12개 고정 후보**를 약한 순서부터 | gain이 이산값 → 약한 지문 가능성 |

CT의 나머지 4종(`acquisition_motion`, `insufficient_projection_sampling`, `low_signal_noise`, `beam_hardening_metal_streak`)과
RGB `trigger_timing_failure`·`surface_dust`·`hair_contamination`은 v1.7과 동일하다.

### 1.4 ✅ (해소) 쿼터 불일치 — 총량은 사양과 정확히 일치

이전 버전 계획에서 "v1.7 쿼터와 실측이 불일치"라고 적었으나, **케이스별 총량은 사양과 완전히 일치**한다.
어긋나는 것은 main/test 배분뿐이다.

| CT 케이스 | main(실측) | test(실측) | 합 | v1.9 §1.2 전체 |
|---|---|---|---|---|
| `cell_alignment_failure` | 380 | 20 | **400** | 400 ✅ |
| `acquisition_motion` | 373 | 27 | **400** | 400 ✅ |
| `insufficient_projection_sampling` | 387 | 13 | **400** | 400 ✅ |
| `low_signal_noise` | 375 | 25 | **400** | 400 ✅ |
| `beam_hardening_metal_streak` | 385 | 15 | **400** | 400 ✅ |

RGB 8종도 동일하게 230/270/270/310/270/270/190/190 전량 일치.

원인: v1.9 §1.2 표의 `test` 열(케이스당 20)은 **구현이 강제하지 않는 값**이다.
§4.2 config에는 `ct_test_fail_target: 100`과 케이스별 *총* 쿼터만 있고,
§4.9 최종 검증도 `CT FAIL case quotas: 400×5`와 `CT test: PASS 900/FAIL 100`만 확인한다.
→ **문서-구현 간 경미한 서술 불일치. 우리는 어차피 재분할하므로 무영향.**
→ 이전에 "v1.8 변경점 문서 필요"로 올린 항목은 v1.9 확보로 **해소**.

### 1.5 도메인 커버리지 제한 (기록용)

- CT는 pouch만, RGB는 cylindrical만. 다른 form에 대한 일반화는 이 데이터로 검증 불가.
- v1.9 §1.1 + §4.2 `max_outputs_per_source: 1` → **동일 원본은 40,000장 중 딱 한 번만 사용**.
  즉 같은 원본의 PASS판/FAIL판 쌍이 존재하지 않으므로 **paired 비교(같은 사진의 전/후)는 불가능**하다.
  다행히 배터리 단위로는 PASS·FAIL이 섞여 있다(CT 47/47, RGB 692/757 배터리가 FAIL 보유)
  → 셀 단위 분할을 해도 각 split에 FAIL이 들어간다.

### 1.6 우리가 받은 것은 전송용 아카이브라 manifest가 빠져 있다

v1.9 §3.2/§4.11에 따르면 정식 산출물 루트에는 `manifests/`, `logs/`, `generation_summary.json`이 있고,
업로드 3단계의 `<Output명>_images.zip`은 **`CT/`·`RGB/` 트리만** 담는다.
우리가 받은 `quality_fail_40k_v1.8_20260730_unzip`이 정확히 그 전송 ZIP을 푼 형태다(manifest 없음).

→ 추가로 받으면 가치가 큰 파일: **§5 요청 목록** 참조.

---

## 2. 🔴 v1.9 §2가 드러낸 합성 지문 후보 — 이번 반영의 최대 수확

설계문서 §3.2가 경고한 "개념이 아니라 지문을 학습" 리스크가, v1.9 사양을 읽으니 **구체적 항목으로 특정**된다.
아래는 추측이 아니라 사양에 명시된 결정론적 처리에서 직접 도출한 것이다.
**Phase 1에서 통계로 존재를 확인하고, Phase 5에서 모델이 실제로 이걸 쓰는지 검증한다.**

| # | 케이스 | 사양상의 결정론적 처리 | 남는 지문 | 위험도 |
|---|---|---|---|---|
| **F1** | `rgb_overexposure` | outline 내부 상위 픽셀을 목표 개수만큼 **정확히 255**로, **나머지 outline 픽셀은 249 이하로 clamp** (bloom 후 재적용) | outline 내부 luminance 히스토그램에 **250~254 구간의 인공적 빈 구멍**. 실제 과노출 사진에는 없는 특징 | **최고** |
| **F2** | `rgb_reflection_glare` | 반사 색을 따뜻한 흰색 **`(255,246,224)` 고정값**으로 alpha 합성 | 특정 색상비(R>G>B, 고정 비율)의 고휘도 화소 클러스터. 단일 색상 필터로 검출 가능 | **최고** |
| **F3** | `rgb_underexposure` | outline 평균비를 `0.44~0.56` 목표로 보정, gate가 `0.40~0.70` + full-frame `≤0.72` 강제 | 밝기 저하량이 **좁은 구간에만** 분포. 실제 저노출의 넓은 분포와 다름 | 높음 |
| **F4** | `rgb_focus_failure` | motion blur 제거, **등방 Gaussian defocus 단일**, gate가 edge energy 비를 `0.25~0.75`로 강제 | blur 커널이 항상 등방 Gaussian. 실제 초점 이탈의 비대칭/필드 곡률 없음 | 높음 |
| **F5** | `rgb_uneven_lighting` | `(dark,bright)` gain이 **12개 고정 후보** 중 하나, 방향은 4방향 중 하나, smoothstep 고정 | 비대칭도가 이산적으로 뭉침 | 중 |
| **F6** | `ct_cell_alignment_failure` | edge crop 후 **LANCZOS 업스케일**로 원본 raster 복원 | LANCZOS 특유의 링잉/주파수 특성. FAIL에만 존재하는 리샘플 흔적 | 중~높음 |
| **F7** | 공통 | PASS는 원본→512 다운스케일 1회, FAIL은 증강 후 512 다운스케일 | 증강 연산이 남기는 노이즈/주파수 통계 차이 | 중 |
| **F8** | 공통 | 전 이미지 JPEG q90 / subsampling 0 / optimize·progressive off로 동일 저장 | **동일 조건이라 지문 없음** — 이 축은 안전 (Phase 1에서 확인만) | 낮음 |

**Phase 1 지문 사전 점검(구체 항목)**
- F1: PASS/FAIL의 outline 내부 250~254 픽셀 비율 히스토그램 비교. FAIL에서 0에 수렴하면 확정
- F2: 고휘도 화소의 `(R-G, G-B)` 산점도. `(255,246,224)` 근방 클러스터 유무
- F3/F5: outline 평균 luminance 비·비대칭도의 **분포 모양**(연속 vs 이산 뭉침)
- F6: FAIL/PASS의 고주파 스펙트럼 기울기 비교 (CT alignment만 따로)

**Phase 5 지문 정량 검증(추가 실험)**
- **F1 마스킹 테스트**: overexposure FAIL의 255 픽셀을 254로 1단계 낮춰 재평가. recall이 급락하면 모델이 F1을 쓰고 있음
- **F2 색 중립화**: glare 영역을 무채색으로 치환 후 재평가
- **shortcut-only 분류기**: F1·F2 두 개 스칼라 피처만으로 로지스틱 회귀. 이것만으로 높은 recall이 나오면 그 케이스의 "성능"은 전부 지문이다

이 8개 중 어느 것도 실촬영 FAIL에는 존재하지 않는다.
→ **§5-4 실촬영 나쁜 샷 확보의 우선순위가 v1.9 반영으로 더 올라갔다.**

---

## 2-a. 규칙 baseline의 기대치가 올라갔다 (Phase 2 중요도 상승)

v1.9 사양상 아래 4종은 **전역 광도/선명도 통계로 거의 결정적으로 분리된다**:

| 케이스 | RGB main 장수 | 결정적 신호 |
|---|---|---|
| `underexposure` | 255 | outline 평균 luminance 비 0.40~0.70 |
| `overexposure` | 254 | outline 포화율 0.15~0.75 |
| `uneven_lighting` | 259 | outline asymmetry 0.25~0.60 |
| `focus_failure` | 299 | edge energy 비 0.25~0.75 |
| **소계** | **1,067 / 1,900 = 56%** | |

각 케이스의 **자동 통과 조건 자체가 곧 판별 규칙**이다. 게이트를 통과한 이미지만 데이터셋에 남았으므로,
그 조건식을 그대로 피처로 쓰면 해당 케이스는 사실상 100%에 가깝게 잡힌다.
→ Phase 2 규칙 baseline은 "혹시 되나 보는 것"이 아니라 **RGB FAIL의 절반 이상을 확실히 커버하는 하한선**이다.
→ MobileNet의 존재 이유는 나머지 44%(`trigger_timing` 218, `glare` 251, `dust` 180, `hair` 185)에서 나온다.
   특히 `surface_dust`(core 지름 최종 긴 변의 1~6%, alpha 0.05~0.20)와 `hair`(두께 0.15~0.6%, alpha 0.05~0.18)는
   **저대비·미세**라 입력 해상도에 민감하다 → Phase 3 입력 크기 sweep의 판정 기준을 이 두 케이스로 잡는다.

---

## 3. 단계별 계획

### ⚡ Fast track — 2026-07-31 오전 (실행됨) · **CT 전담**

담당 범위가 **CT 전용**으로 확정됐다. RGB 관련 산출물(캐시·런)은 제거했고
`src/*`는 CT 기준으로 재작성했다. RGB 코드가 필요해지면 git 이력에서 복원한다.

**압축한 것은 순서가 아니라 범위다.** 셀 누수 방지(battery_id 분할)와 지표 정의는
비용이 0에 가까우면서 결과의 유효성을 좌우하므로 그대로 넣었고,
지문 검증·5-fold 전체·입력크기 sweep은 뒤로 미뤘다.

| 단계 | 상태 | 소요 |
|---|---|---|
| venv + torch 2.13.0+cpu / torchvision 0.28.0+cpu | ✅ | ~3분 |
| 인덱스 20,000행(CT) + 라벨 3중 대조 (불일치 0건) | ✅ | ~4분 |
| battery_id 재분할 — lock-box 9셀 봉인 + 38셀 5-fold | ✅ | 즉시 |
| **CT 형상 실측** → 캔버스 288×512 결정 | ✅ | ~5분 |
| CT letterbox 캐시 288×512 (8.8 GB memmap) | ✅ | 32초 |
| MobileNetV3-Small 학습 (fold0, 8 epoch) | ✅ 실행 | ~4분/epoch |
| `predict.py` 폴더 분류 → CSV | ✅ | — |

**CT 전용 설계 결정 3건 (README에 근거 포함)**
1. **캔버스 288×512** — height는 20,000장 전부 512 고정이고 width만 46~282.
   정사각 512로 맞추면 평균 74%가 패딩이다. 288이면 화소 손실 0에 연산 0.56배.
2. **종횡비 shortcut 없음(확인함)** — `ct_cell_alignment_failure`가 LANCZOS로
   원본 raster에 되돌리므로 PASS·FAIL의 w/h 분포가 동일(p50 둘 다 0.2891).
3. **flip은 hflip+vflip 둘 다** — CT ROI는 상하 대칭도 물리적으로 유효.
   광도·블러 증강은 여전히 금지(그 자체가 `low_signal_noise`·`acquisition_motion` 신호).

**오늘 미룬 것 (원래 Phase 순서대로 이어서 진행)**
- 5-fold 나머지 4개 (`src/run_ct_cv.py` 작성 완료 — fold0 결과 확인 후 실행)
- Phase 1의 PASS 오염 스캔, F6 지문 사전 점검
- Phase 2 규칙 baseline (`src/ct_rule_baseline.py` 작성 완료, 실행 대기)
- Phase 3 입력 크기 sweep (오늘은 288×512 단일)
- Phase 4 셀 단위 집계, Phase 5 지문 절제, Phase 6 체인 검증
- **lockbox 9셀은 미개봉 유지**

**clean FPR 상한이 미정이라 블로킹하지 않고** 1%/3%/5% 세 지점을 모두 출력하도록 했다.
상한이 정해지면 `metrics_val.json`에서 해당 threshold만 읽어 쓰면 되고 재학습은 불필요하다.

---

### Phase 0 — 환경 구축 + 데이터 인덱싱  *(0.5일)*

목표: 재현 가능한 환경과, 40,000행 단일 인덱스.

1. `py -3.14 -m venv .venv` → `torch==2.13.0 torchvision==0.28.0`(CPU), `numpy pandas pillow scikit-learn matplotlib tqdm`
2. 프로젝트 스캐폴드
   ```
   quality_gate_mobilenet/
   ├─ src/           dataset.py, model.py, transforms.py, metrics.py, rules.py
   ├─ notebooks/     00_index.ipynb, 01_audit_split.ipynb, 02_rule_baseline.ipynb, 03_train.ipynb, 04_eval.ipynb
   ├─ manifests/     index_ct.csv, index_rgb.csv
   ├─ splits/        ct_folds.csv, rgb_split.csv
   ├─ cache/         letterbox uint8 memmap
   └─ runs/          체크포인트·로그·지표
   ```
3. `manifests/index_{ct,rgb}.csv` 생성 — 열: `path, modality, orig_partition, quality_class, failure_case, battery_id, axis, form, width, height, image_id`
   - `quality_class`는 **`labels_json`에서** 읽고, `pass/`·`fail/` 폴더 및 `augmentation_json` 존재 여부와 **3중 대조**해 불일치 0건 확인
4. 무결성 체크: 이미지↔JSON 1:1, 중복 image_id 0, decode 실패 0

**완료 기준**: `import torch` 성공, 인덱스 40,000행, 대조 불일치 0.

---

### Phase 1 — 감사 & 재분할  *(1일, 가장 중요)*

1. **누수 재현·기록** — §1.1 수치를 스크립트로 재생성해 리포트에 박아둔다(제공 test 폐기 근거).
2. **battery_id 기준 재분할**
   - **RGB (757 셀)**: 셀 단위 train/val/test = 60/20/20.
     셀별 FAIL 비율과 실패 케이스 구성으로 층화. 목표는 각 split이 8개 케이스를 모두 포함.
   - **CT (47 셀)**: 단일 split 금지.
     셀 단위 **GroupKFold(5)** → 각 fold ≈ 9~10셀. 전 지표를 5-fold 평균±표준편차로 보고.
     최종 확인용으로 8~10셀을 완전 봉인(lock-box)하고 마지막 1회만 개봉.
   - 검증: split 간 battery_id 교집합 0, 각 split의 케이스별 최소 장수 기록.
3. **PASS 오염 스캔** (설계문서 §3.1 리스크)
   - 라플라시안 분산(흐림), 평균/표준편차 luminance, 포화 픽셀 비율, 고주파 에너지
   - PASS 분포의 극단 꼬리를 FAIL 분포와 겹쳐 그려 "PASS인데 FAIL 같은" 후보 목록 산출
   - 심한 것은 제외 리스트로 관리(삭제 X, 플래그만)
4. **합성 지문 사전 점검 (§2의 F1~F8을 그대로 측정)** — v1.9 반영으로 항목이 구체화됨
   - **F1**: outline 내부 250~254 픽셀 비율 — PASS vs `rgb_overexposure` FAIL 히스토그램. FAIL이 0에 수렴하면 지문 확정
   - **F2**: 고휘도 화소의 `(R−G, G−B)` 산점도 — `(255,246,224)` 근방 클러스터 유무
   - **F3/F5**: outline 평균 luminance 비·비대칭도의 **분포 모양** (연속인가, 이산으로 뭉치는가)
   - **F6**: `ct_cell_alignment_failure`만 따로 고주파 스펙트럼 기울기를 PASS와 비교 (LANCZOS 업스케일 흔적)
   - **F8**: JPEG 파일 크기·양자화 테이블 분포 비교 — 사양상 전 이미지 동일 설정이므로 **차이 없음이 정상**. 있으면 그게 이상
   - 산출: `runs/fingerprint_audit.md` — 각 항목 통과/의심 판정과 근거 그림

**완료 기준**: `splits/*.csv` 확정(교집합 0), 오염 후보 리스트, F1~F8 지문 점검 리포트.

---

### Phase 2 — 규칙 baseline  *(1일)* — MobileNet **이전에** 반드시

설계문서 §6-2의 지적: 합성 FAIL이 전역 왜곡 위주라 규칙만으로 상당히 잡힐 수 있다.
**v1.9 §2를 읽고 나서 이 지적의 근거가 확정됐다(§2-a): RGB FAIL의 56%는 각 케이스의 자동 통과 조건식이 곧 판별식이다.**
이 단계의 목적은 셋 — **하한선 확보**, **평가 파이프라인 선(先)확정**, **지문 기여도 정량화**.

1. 지표 모듈부터 고정: `fail recall @ clean FPR ≤ τ`, PR-AUC, 케이스별 recall, threshold sweep
   - accuracy 금지 (90:10에서 "전부 pass"가 90%)
   - clean FPR = 정상 이미지를 FAIL로 = 재촬영 = 수율 손실
2. **피처를 v1.9 자동 통과 조건에서 직접 유도** (임의 설계가 아니라 사양 역산)
   - `outline 평균 luminance 비` → `underexposure` (§2.3.5 gate: 0.40~0.70)
   - `outline 내부 포화율` → `overexposure` (§2.3.6 gate: 0.15~0.75)
   - `축 양 끝 20% 평균차 / outline 평균` → `uneven_lighting` (§2.3.2 gate: 0.25~0.60)
   - `RMS gradient edge energy 비` → `focus_failure` (§2.3.4 gate: 0.25~0.75)
   - `outline 잔여 면적비 + 프레임 경계 접촉` → `trigger_timing`, CT `cell_alignment` (gate: 0.50~0.90)
   - `균일영역 noise σ`, `edge SNR 비` → CT `low_signal_noise`
   - `방향성 streak 에너지`, `고주파 edge energy 감소율` → CT `insufficient_projection_sampling`
   - 보조: 라플라시안 분산, 밝기 왜도, 암부율, 색 채도 분산
   - ※ battery outline은 label JSON의 `swelling.battery_outline`을 rasterize해 사용 (증강계획서와 동일 기준)
3. 로지스틱 회귀 + GBM 2종 → 케이스별 recall 표
4. **지문 기여도 분리(신규)**: §2의 F1·F2 스칼라 2개만으로 만든 "shortcut-only" 분류기를 나란히 평가.
   이 값이 전체 baseline과 비슷하면 그 성능은 개념이 아니라 지문이다 — 숫자를 신뢰하지 말 것
5. **판정**: 규칙만으로 목표를 만족하면 MobileNet 생략 가능.
   못 잡는 케이스를 명시(예상: `surface_dust`, `hair_contamination`, `reflection_glare` 일부)

**완료 기준**: 규칙 baseline의 `fail recall @ clean FPR` 곡선, 케이스별 recall 표, shortcut-only 대조값.

---

### Phase 3 — MobileNetV3-Small 학습  *(2~3일)*

1. **전처리 캐시** — letterbox 결과를 uint8 memmap으로 1회 생성(CPU 학습에서 JPEG 디코드 반복 제거)
   - 224: 3.0 GB/모달, 384: 8.8 GB/모달 (디스크 66 GB 여유 내)
2. **모델** — torchvision `mobilenet_v3_small(IMAGENET1K_V1)`, `classifier[3]` → `Linear(1024, 1)`, CT·RGB 각 1개
   - CT 흑백은 3채널 복제 후 ImageNet mean/std 정규화
3. **letterbox 필수, crop 금지** — pad 값은 PASS/FAIL 동일하게(shortcut 차단)
4. **입력 크기 sweep** — RGB 224 → 320 → 384, CT 256 → 384 → 512.
   판정 기준은 전체 recall이 아니라 **미세·저대비 케이스** recall. v1.9 사양 기준 해당 케이스와 그 크기:
   - `rgb_surface_dust`: core 지름 = 최종 긴 변의 **1~6%** (512 기준 5~31px), core alpha 0.05~0.20 + 흐린 halo
   - `rgb_hair_contamination`: 중심선 두께 = 최종 긴 변의 **0.15~0.6%** (512 기준 0.8~3px), alpha 0.05~0.18
   → hair는 224로 줄이면 두께가 **0.3~1.3px**가 되어 소실 위험. **RGB는 384 이상이 사실상 필수**로 보이며, sweep은 이를 확인하는 절차
   - CT는 `insufficient_projection_sampling`(고주파 손실)과 `acquisition_motion`(512 공간 18~28px ghost offset) 기준
5. **불균형** — `BCEWithLogitsLoss(pos_weight=9.0)`. WeightedRandomSampler는 대안으로만
6. **학습 증강은 flip만.** 밝기·블러·노이즈 증강 금지 — 그 자체가 FAIL 신호라 PASS에 걸면 라벨 오염
7. AdamW, cosine, 12~20 epoch, early stop on `fail recall @ clean FPR`
   - CPU 예산: 224 기준 에폭당 수 분, 512 기준 10분 내외 예상. 1회 실측 후 sweep 범위 확정

**완료 기준**: 모달별 최적 입력 크기 확정, val 지표 기록, 체크포인트 저장.

---

### Phase 4 — 평가와 운영점 결정  *(0.5일)*

1. threshold sweep → ROC/PR, **clean FPR 상한을 먼저 정하고 그 안에서 fail recall 최대**인 지점 선택
2. 케이스별 recall 표 (어떤 촬영 실패를 놓치는지 = 현장 리스크 목록)
3. CT는 5-fold 평균±표준편차로 보고 (47셀이라 단일 숫자는 신뢰 불가)
4. **셀 단위 집계** — 실제 의사결정은 이미지가 아니라 셀 단위. 셀의 N장 중 k장 FAIL이면 재촬영, 같은 규칙의 운영 지표도 함께 산출
5. 오분류 갤러리 — FN/FP 상위 50장 육안 확인

---

### Phase 5 — 합성 함정 정면 돌파  *(1일)* — 설계문서 §3.2

숫자가 좋게 나온 뒤에 반드시 하는 단계. 안 하면 "합성 0.99, 실배포 붕괴".

1. **Leave-one-case-out** — 케이스 1종을 학습에서 완전히 빼고 그 케이스 recall 측정.
   무너지면 모델이 "촬영 품질"이 아니라 "그 증강 함수의 지문"을 배운 것.
2. **지문 절제(ablation) 테스트 — v1.9 반영 신규**. §2의 F1·F2가 사양에 명시된 결정론적 처리이므로 직접 무력화해 볼 수 있다.
   - **F1**: `rgb_overexposure` FAIL의 값 255 픽셀을 254로 1단계 낮춰 재평가.
     recall이 급락하면 모델은 "과노출"이 아니라 **"255와 249 사이의 인공적 히스토그램 구멍"** 을 배운 것
   - **F2**: glare 영역 화소를 같은 luminance의 무채색으로 치환 후 재평가.
     recall이 급락하면 **고정색 `(255,246,224)`** 를 배운 것
   - **F6**: `ct_cell_alignment_failure`에 약한 저역통과를 걸어 LANCZOS 링잉을 흐린 뒤 재평가
   - 판정: 위 조작으로 recall이 20%p 이상 떨어지는 케이스는 **실배포 성능을 0으로 간주**하고 별도 보고
3. **재압축 견고성** — test 이미지를 JPEG q75/q95로 재저장 후 지표 변화 측정
4. **PASS 오염 재확인** — Phase 1 후보를 제외했을 때 지표 변화
5. **실촬영 나쁜 샷 확보 요청**(데이터팀) → 합성-실제 gap 측정.
   **이게 없으면 어떤 숫자도 실배포 성능의 근거가 되지 않는다.**
   v1.9 사양을 읽고 나서 이 항목의 우선순위가 더 올라갔다 — §2의 F1~F7 중 어느 것도 실촬영 FAIL에는 존재하지 않기 때문이다.

---

### Phase 6 — 체인 검증 & shadow 배포  *(별도 합의 필요)*

1. **end-to-end**: held-out에서 `게이트 pass-subset의 YOLO F1` vs `게이트 없는 전체 F1`.
   pass-subset ≥ 전체여야 게이트 도입이 정당.
2. **latency 실측**: 100% 이미지에 태우므로 CT는 셀당 슬라이스 수만큼 곱해짐. 셀당 예산 대비 1회 측정.
3. **shadow 모드**: 처음엔 로그만 남기고 실제 필터링 X → 실 clean FPR 확인 후 활성화.

---

## 4. 권장 진행 순서

**RGB 먼저.** 757셀로 셀 단위 분할이 정상 작동하고, 케이스 8종·해상도 고정(512×288)이라 파이프라인 검증이 빠르다.
여기서 Phase 0~4를 한 바퀴 완주해 코드와 지표를 굳힌 뒤, 동일 코드로 CT(47셀·가변 해상도·GroupKFold)를 돌린다.

## 5. 착수 전 확정이 필요한 값 / 추가 확보 요청

**확정 필요**
1. **clean FPR 상한** — 정상 이미지 몇 %까지 재촬영을 감수할 수 있나? (예: 1%, 3%, 5%) 운영점과 목표 recall이 여기서 결정된다. ← **유일한 블로커**
2. **CT lock-box 크기** — 47셀 중 몇 셀을 최종 확인용으로 봉인할지 (권장 8~10셀).
3. **실촬영 FAIL 확보 가능 여부/일정** — Phase 5의 전제. v1.9 반영으로 우선순위 상승.

**추가로 받으면 가치가 큰 파일** (§1.6 — 전송 ZIP에는 안 들어 있음)

| 파일 | 쓸모 | 필요도 |
|---|---|---|
| `manifests/lineage_private.csv` | `actual_parameters_json`에 케이스별 **실제 severity·파라미터**가 있음 → **"약한 FAIL을 놓치는가"** 를 severity별 recall 곡선으로 분석 가능. 지금은 4,000개 JSON에서 일부만 역산 가능 | **높음** |
| `manifests/dataset_manifest.csv` | `failure_case`·해상도·SHA-256을 JSON 4,000개 파싱 없이 즉시 확보. 무결성 대조 | 중 |
| `manifests/generation_errors.csv` + `recovery_audit.csv` | reserve 승계가 일어난 slot 목록 → 특정 케이스에서 원본 pool이 편중됐는지 확인 | 중 |
| `generation_summary.json` | 생성 시점 config hash·검증 수치. 재현성 기록 | 낮음 |

※ `lineage_private.csv`는 원본 절대경로·원본 ID를 포함하는 비공개 파일이므로,
   필요한 것은 `synthetic_id`, `failure_case`, `item_seed`, `actual_parameters_json` 네 열뿐이다.
   원본 경로 열을 제거한 사본으로 받아도 목적을 100% 달성한다.

**해소된 항목**
- ~~v1.8 변경점 문서~~ → v1.9 확보로 해소(§1.4). v1.9 §2 = v1.8 알고리즘 = 우리 데이터의 정본 사양.
