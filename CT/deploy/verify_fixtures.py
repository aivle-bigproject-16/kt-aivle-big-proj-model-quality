# -*- coding: utf-8 -*-
"""골든 fixture 검증 — 이 스크립트가 통과하면 전처리가 학습과 동일하다.

사용법:
    python verify_fixtures.py              # 옆에 있는 quality_ct.pt 를 검사한다
    python verify_fixtures.py <가중치.pt>   # 다른 파일을 검사할 때

가중치는 model_card.json 의 identity.weight_sha256 과 대조한다.
필요한 것은 torch, torchvision, pillow, numpy 뿐이다.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
from PIL import Image

HERE = Path(__file__).parent
CANVAS = (288, 512)          # (W, H) — 비정사각이다
PAD = 114
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox(img: Image.Image, canvas=CANVAS) -> Image.Image:
    cw, ch = canvas
    w, h = img.size
    s = min(cw / w, ch / h)                       # 확대하지 않는다
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    img = img.resize((nw, nh), Image.BILINEAR)
    out = Image.new(img.mode, (cw, ch), (PAD,) * len(img.getbands()))
    out.paste(img, ((cw - nw) // 2, (ch - nh) // 2))
    return out


def preprocess(path: Path) -> torch.Tensor:
    a = np.array(letterbox(Image.open(path).convert("L")).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    return (x - MEAN) / STD


def load_model(path: Path) -> nn.Module:
    sd = torch.load(path, map_location="cpu")
    if isinstance(sd, dict):
        sd = sd.get("model", sd.get("state_dict", sd))
    m = tv.models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    m.load_state_dict(sd)
    m.eval()
    return m


def main(weight: Path) -> int:
    spec = json.loads((HERE / "fixtures" / "expected.json").read_text(encoding="utf-8"))
    tol = spec["tolerance"]["logit_abs"]
    thr = spec["model"]["threshold"]

    sha = hashlib.sha256(weight.read_bytes()).hexdigest()
    want = spec["model"]["weight_sha256"]
    print(f"가중치 {weight}")
    print(f"  sha256 {sha}")
    if sha != want:
        print(f"  [!] fixture 가 기대하는 가중치가 아니다 (기대 {want})")
        print("      다른 학습본이면 logit 이 당연히 다르다. 먼저 가중치를 맞추십시오.")
        return 2
    print("  sha256 일치")

    model = load_model(weight)
    bad = []
    with torch.no_grad():
        for c in spec["cases"]:
            got = float(model(preprocess(HERE / "fixtures" / c["file"])).item())
            d = abs(got - c["expected"]["logit"])
            if d > tol:
                bad.append((c["file"], c["expected"]["logit"], got, d))

    n = len(spec["cases"])
    if bad:
        print(f"\n실패 {len(bad)}/{n} — 전처리가 학습과 다르다")
        print(f"  {'파일':<46}{'기대':>12}{'실제':>12}{'차이':>12}")
        for f, e, g, d in bad[:10]:
            print(f"  {f:<46}{e:>12.6f}{g:>12.6f}{d:>12.6f}")
        print("\n  흔한 원인: 정사각 리사이즈 / 흑백 변환 누락 / ImageNet 정규화 누락 / logit 에 sigmoid")
        return 1

    print(f"\n통과 {n}/{n}  (|Δlogit| <= {tol})")
    print(f"  참고: label 은 threshold {thr:+.6f} 기준이며 배포 데이터에서 재보정이 필요하다")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 2:
        print(__doc__)
        sys.exit(2)
    w = Path(sys.argv[1]) if len(sys.argv) == 2 else HERE / "quality_ct.pt"
    if not w.exists():
        print(f"가중치를 찾을 수 없다: {w}")
        sys.exit(2)
    sys.exit(main(w))
