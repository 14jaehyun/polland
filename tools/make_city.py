"""도시 그림 한 장 → city/<id>.jpg + city/<id>.json (깊이 격자, 그림 속 위 방향, 틀) + cities.json 목록 갱신

사용: python3 tools/make_city.py 그림.png shinjuku "Shinjuku" "新宿 · 도쿄" [--adj 1] [--model vits.onnx]
깊이 모델: Depth Anything V2 small ONNX (github fabio-sim/Depth-Anything-ONNX v2.0.0, depth_anything_v2_vits.onnx)
틀은 그림 비율로 자동 판별:
  2:3      → 기준 틀 전체 (예전 그림)
  9:16     → 가이드 1판 (u 0.247~0.863, v 0.15~0.88)
  1200:2450→ 가이드 2판 (사용 영역 1200×2450 + 사방 10% 버퍼)
"""
import sys, json, base64, argparse, os
import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument('img'); ap.add_argument('id'); ap.add_argument('name'); ap.add_argument('sub')
ap.add_argument('--adj', type=float, default=0); ap.add_argument('--model', default='/tmp/claude-0/da/vits.onnx')
ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), '..', 'city'))
a = ap.parse_args()

im = Image.open(a.img).convert('RGB'); W, H = im.size; r = W / H
FRAMES = {'base': (0, 1, 0, 1, 2/3), 'guide1': (.247, .616, .15, .73, 9/16), 'guide2': (.26738, .57024, .1233, .7764, 1200/2450)}
fk = min(FRAMES, key=lambda k: abs(FRAMES[k][4] - r)); U0, UW, V0, VH, _ = FRAMES[fk]
print('frame', fk, (W, H))

import onnxruntime as ort
x = np.asarray(im.resize((518, 518), Image.BICUBIC), dtype=np.float32) / 255
x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None].astype(np.float32)
s = ort.InferenceSession(a.model)
o = np.squeeze(s.run(None, {s.get_inputs()[0].name: x})[0])
d = np.asarray(Image.fromarray(o.astype(np.float32)).resize((W, H), Image.BICUBIC)); d = (d - d.min()) / (d.max() - d.min())

# 시차 → 시야 깊이 (몽콕 그림에서 맞춘 값)
A, B = 0.009259666907146335, 0.002334771897011315
NX = 145
NY = int(round((NX - 1) * H / W)) + 1
ys = np.linspace(0, H - 1, NY).round().astype(int); xs = np.linspace(0, W - 1, NX).round().astype(int)
vd = (1 / (A * d[np.ix_(ys, xs)] + B)).astype('<f4')

# 그림 속 위 방향: 띄운 면들 중 위를 보는 면(옥상·바닥)의 평균 법선
D = np.pi / 180
def Rx(t): c, s_ = np.cos(t), np.sin(t); return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])
def Ry(t): c, s_ = np.cos(t), np.sin(t); return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
R = Ry(-28 * D) @ Rx(-56 * D); tv = np.tan(37 * D); asp = 2 / 3
P = np.array([-5.14793423, 339.68815258, 4.92460418])   # 기준 카메라 위치
u = U0 + UW * np.linspace(0, 1, NX)[None, :]; v = V0 + VH * np.linspace(0, 1, NY)[:, None]
pw = np.stack([(u * 2 - 1) * asp * tv * vd, (1 - 2 * v) * tv * vd, -vd + 0 * u], -1) @ R.T + P
e1 = pw[:-1, 1:] - pw[:-1, :-1]; e2 = pw[1:, :-1] - pw[:-1, :-1]
n = np.cross(e2, e1); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9; n = n.reshape(-1, 3)
n[(n * (P - pw[:-1, :-1].reshape(-1, 3))).sum(1) < 0] *= -1
up = np.array([0, 1., 0])
for it in range(8):
    m = n @ up > np.cos((30 if it < 3 else 15) * D); up = n[m].mean(0); up /= np.linalg.norm(up)
print('up', up.round(4))

os.makedirs(a.out, exist_ok=True)
im.save(os.path.join(a.out, a.id + '.jpg'), 'JPEG', quality=90)
json.dump({'id': a.id, 'name': a.name, 'sub': a.sub, 'nx': NX, 'ny': NY, 'vd': base64.b64encode(vd.tobytes()).decode(),
           'up': [round(float(t), 5) for t in up], 'adj': a.adj, 'frame': [U0, UW, V0, VH]},
          open(os.path.join(a.out, a.id + '.json'), 'w'))
lp = os.path.join(a.out, 'cities.json')
L = json.load(open(lp)) if os.path.exists(lp) else []
L = [c for c in L if c['id'] != a.id] + [{'id': a.id, 'name': a.name}]
json.dump(L, open(lp, 'w'), ensure_ascii=False, indent=1)
print('ok', a.id)
