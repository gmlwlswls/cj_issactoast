# Isaac Sim 5.1 Palletizing Simulator

본 시뮬레이터는 Isaac Sim 5.1 기반의
3D 팔레타이징 물리 검증 환경입니다.

참가자는 자신의 알고리즘 결과(JSON)를
Isaac Sim 환경에서 검증할 수 있습니다.

공식 평가는 대회 운영 서버의 고정된 Isaac Sim 5.1 환경에서 수행됩니다.

---

# 시뮬레이터 목적

본 시뮬레이터는 다음 항목을 검증하기 위해 사용됩니다.

* 박스 충돌 여부
* 팔레트 영역 이탈 여부
* 박스 안정성
* 물리 붕괴 여부
* 최종 적재 성공 여부

---

# Isaac Sim 5.1 설치

NVIDIA 공식 홈페이지에서
Isaac Sim 5.1을 설치합니다.

지원 방식:

* Local 설치
* Docker 설치

NVIDIA 공식 설치 문서를 참고하세요.

---

# 시뮬레이터 코드 복사

제공된 `palletizing_simulator`
폴더를 Isaac Sim 내부에 복사합니다.

예시:

```text
/isaac-sim/palletizing_simulator/
```

최종 구조 예시:

```text
/isaac-sim
├── python.sh
├── kit/
├── exts/
└── palletizing_simulator/
    ├── algorithm_results/
    ├── box_sequence/
    ├── config/
    │   └── simulator_config.yaml
    ├── sim_results/
    ├── buffer_manager.py
    ├── evaluator.py
    ├── monitor.py
    ├── scene.py
    ├── simulator.py
    └── README.md
```

---

# 실행 방법

Isaac Sim 설치 경로에서 아래 명령을 실행합니다.

```bash
cd /isaac-sim

./python.sh palletizing_simulator/simulator.py
```

실행 시:

* Isaac Sim 실행
* simulator.py 실행
* 입력 JSON 로드
* 물리 시뮬레이션 수행

이 자동으로 수행됩니다.

---

# 실행 로그 예시

정상 실행 시 예시:

```text
[INFO] simulator started
[INFO] loading algorithm_results
[INFO] loading box_sequence
[INFO] simulation started
...
```

---

# 경로 설정

시뮬레이터는
`config/simulator_config.yaml`의 `paths` 항목을 사용합니다.

```yaml
paths:
  input_dir: "/isaac-sim/palletizing_simulator/algorithm_results"
  box_sequence_dir: "/isaac-sim/palletizing_simulator/box_sequence"
  output_dir: "/isaac-sim/palletizing_simulator/sim_results"
```

각 경로의 의미는 다음과 같습니다.

| 항목               | 설명                               |
| ---------------- | -------------------------------- |
| input_dir        | 참가자의 알고리즘 실행 결과 JSON 파일이 저장되는 폴더 |
| box_sequence_dir | 알고리즘 입력 박스 시퀀스 JSON 파일이 저장되는 폴더  |
| output_dir       | Isaac Sim 물리 시뮬레이션 결과가 저장되는 폴더   |

---

# 입력 파일 매칭 규칙

`input_dir` 와 `box_sequence_dir`
내부의 파일 이름은 반드시 동일해야 합니다.

예시:

```text
algorithm_results/
├── sample1.json
├── sample2.json

box_sequence/
├── sample1.json
├── sample2.json
```

시뮬레이터는 동일한 파일 이름을 기준으로:

* 참가자 알고리즘 결과
* 원본 입력 박스 시퀀스

를 매칭하여 시뮬레이션을 수행합니다.

따라서 파일 이름이 다르면 정상적으로 평가되지 않을 수 있습니다.

---

# 실행 흐름

시뮬레이터는 다음 순서로 동작합니다.

1. `algorithm_results/` 에서 참가자 알고리즘 결과 JSON 로드
2. `box_sequence/` 에서 입력 박스 시퀀스 로드
3. Isaac Sim 환경에서 박스를 순서대로 생성
4. 물리 안정화(Settling) 수행
5. 충돌 및 붕괴 여부 평가
6. 결과를 `sim_results/` 에 저장

---

# 좌표계

시뮬레이터 좌표계는 다음과 같습니다.

```text
X축: 팔레트 길이 방향
Y축: 팔레트 폭 방향
Z축: 높이 방향
```

원점(origin):

```text
(0, 0, 0)
```

팔레트 바닥 좌측 하단 기준입니다.

---

# 출력 결과

시뮬레이션 결과는 다음 경로에 저장됩니다.

```text
sim_results/
```

예시:

```text
sim_results/
├── result.json
└── screenshot.png
```

---

# 물리 설정

시뮬레이터는 다음 물리 항목을 사용합니다.

* friction
* damping
* restitution
* settling step
* collision check

세부 설정은 아래 파일에서 수정 가능합니다.

```text
config/simulator_config.yaml
```

---

# 주의 사항

* Isaac Sim 5.1 설치가 필요합니다.
* NVIDIA GPU와 호환 가능한 Driver가 필요합니다.
* 참가자 PC 환경에 따라 실행되지 않을 수 있습니다.
* 로컬 시뮬레이션 결과와 공식 평가 서버 결과는 일부 차이가 있을 수 있습니다.
* 공식 평가는 대회 운영 서버 환경 기준으로 수행됩니다.

---

# 공식 평가 환경

공식 평가는 다음 환경 기준으로 수행됩니다.

* Isaac Sim 5.1
* Ubuntu Linux
* NVIDIA GPU
* 대회 운영 서버의 고정된 simulator_config.yaml 설정

참가자는 로컬 환경과 평가 서버 환경 차이가 발생할 수 있음을 고려해야 합니다.
