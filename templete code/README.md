# 3D Palletizing Challenge Baseline

본 프로젝트는 버퍼 기반 3D 팔레타이징(Palletizing) 알고리즘 개발을 위한
대회 참가자용 베이스라인 코드입니다.

참가자는 `algorithm.py` 의 `Palletizer` 클래스를 수정하여
더 높은 적재율과 더 나은 적재 전략을 구현할 수 있습니다.

---

# 프로젝트 구조

```text
project/
├── algorithm.py
├── buffer_manager.py
├── main.py
├── visualize.py
├── requirements.txt
├── config/
│   └── algorithm_config.yaml
├── box_sequence/
│   ├── sample1.json
│   └── sample2.json
└── algorithm_results/
```

| 파일                           | 설명                  |
| ---------------------------- | ------------------- 
| algorithm.py                 | 참가자가 수정하는 핵심 알고리즘   
| buffer_manager.py            | 버퍼 관리 유틸리티 (수정 불필요) 
| main.py                      | 실행 프레임워크            
| visualize.py                 | 적재 결과 시각화           
| requirements.txt             | Python 패키지 목록       
| config/algorithm_config.yaml | 알고리즘 설정             
| box_sequence/                | 입력 박스 시퀀스           
| algorithm_results/           | 결과 저장 디렉토리          

---

# 설치 방법

Python 3.12를 권장합니다.

```bash
pip install -r requirements.txt
```

---

# 실행 방법

```bash
python main.py
```

실행 시:

1. 입력 JSON 파일 로드
2. 팔레타이징 알고리즘 수행
3. 결과 JSON 저장
4. 시각화 PNG 저장
5. 통계 출력

이 자동으로 수행됩니다.

---

# 실행 결과

알고리즘 실행 결과는 다음 경로에 저장됩니다.

```text
algorithm_results/
```

예시:

```text
algorithm_results/
├── sample1.json
├── sample2.json
└── vis/
    ├── sample1.png
    └── sample2.png
```

---

# 참가자 개발 영역

참가자는 주로 아래 파일을 수정하면 됩니다.

```text
algorithm.py
```

핵심 클래스:

```python
class Palletizer:
```

참가자는 자유롭게:

* 새로운 적재 전략 구현
* 탐색 알고리즘 추가
* Helper class/function 추가
* 외부 라이브러리 사용

등을 수행할 수 있습니다.

---

# 수정 금지 항목

아래 구조는 평가 시스템과 연동되므로 수정하지 마세요.

* `BoxInput`
* `PlacedBox`
* `RunResult`
* `PalletConfig`
* `run()` 함수 시그니처

---

# 좌표계

본 프로젝트의 좌표계는 아래와 같습니다.

```text
X축: 팔레트 길이 방향
Y축: 팔레트 폭 방향
Z축: 팔레트 높이 방향
```

원점(origin)은 팔레트 바닥의 좌측 하단 모서리입니다.

```text
origin = (0, 0, 0)
```

팔레트 영역은 다음 범위로 정의됩니다.

```text
0 <= x <= pallet.length
0 <= y <= pallet.width
0 <= z <= pallet.height
```

박스의 `position` 값은 박스의 중심 좌표입니다.

예를 들어 크기가 `[0.3, 0.2, 0.1]` 인 박스를
팔레트 원점에 맞춰 바닥에 놓는 경우:

```text
box size     = [0.3, 0.2, 0.1]
bottom-left  = [0.0, 0.0, 0.0]
position     = [0.15, 0.10, 0.05]
```

---

# 입력 데이터 형식

입력 파일은 JSON 배열 형식입니다.

예시:

```json
[
  {
    "step": 0,
    "id": 1,
    "size": [0.3, 0.2, 0.1],
    "mass": 2.0
  }
]
```

| 필드   | 설명                      |
| ---- | ----------------------- |
| step | 컨베이어 도착 순서              |
| id   | 박스 ID                   |
| size | [length, width, height] |
| mass | 박스 무게                   |

---

# 출력 데이터 형식

알고리즘 결과는 JSON 파일로 저장됩니다.

예시:

```json
{
  "buffer_size": 1,
  "sequence": [
    {
      "step": 0,
      "id": 1,
      "size": [0.3, 0.2, 0.1],
      "mass": 2.0,
      "position": [0.15, 0.1, 0.05],
      "rotation": 0
    }
  ],
  "terminated": false,
  "terminated_step": null
}
```

---

# 출력 규칙

* 모든 `position` 값은 박스 중심 좌표 기준입니다.
* `size` 는 실제 회전이 반영된 크기여야 합니다.
* `rotation` 은 현재 `0` 또는 `90` 만 허용됩니다.
* 모든 단위는 meter(m) 입니다.
* 박스는 팔레트 영역 밖으로 벗어나면 안 됩니다.
* 박스끼리 충돌하면 안 됩니다.
* 박스는 충분한 지지를 받아야 합니다.

---

# 평가 기준

적재율(Utilization)은 아래 기준으로 계산됩니다.

```text
적재율 =
(적재된 박스 총 부피)
/
(팔레트 길이 × 폭 × 높이)
```

현재 기본 팔레트 크기:

```yaml
pallet:
  length: 1.2
  width: 1.0
  height: 1.25
```

---

# 버퍼(Buffer) 개념

```yaml
buffer:
  size: 4
```

예를 들어 buffer size 가 4이면:

* 현재 박스 포함 최대 4개 박스를 동시에 확인 가능
* 참가자는 이 중 어떤 박스를 먼저 적재할지 선택 가능
* 박스 적재 시 자동으로 보충됨

---

# 회전(Rotation)

현재 baseline 은 Z축 기준 90도 회전을 지원합니다.

```python
rotation = 0
rotation = 90
```

회전된 크기는 출력 `size` 에 반영되어야 합니다.

예를 들어 입력 박스 크기가 다음과 같을 때:

```text
original size = [0.3, 0.2, 0.1]
```

90도 회전하면 출력 크기는 다음과 같습니다.

```text
rotated size = [0.2, 0.3, 0.1]
rotation     = 90
```

---

# 시각화

실행 후 결과 PNG 가 저장됩니다.

```text
algorithm_results/vis/
```

예시:

* sample1.png
* sample2.png

---

# 참고 사항

* baseline 코드는 매우 단순한 휴리스틱입니다.
* 참가자는 자유롭게 새로운 탐색 알고리즘을 구현할 수 있습니다.
* 새로운 helper class/function 추가도 가능합니다.
* ONNX Runtime 등 외부 추론 엔진 사용 가능합니다.

---

# 라이선스

본 코드는 대회 참가 목적으로 제공됩니다.
