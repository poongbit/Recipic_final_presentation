# step02: Open Images(검증·테스트 분할)로 독립 테스트셋을 만들고 현재 모델을 평가합니다
import os, io, json, random, shutil, subprocess, urllib.request, concurrent.futures as cf
import pandas as pd
from PIL import Image

G = 'https://storage.googleapis.com/openimages'
S3 = 'https://open-images-dataset.s3.amazonaws.com'
BOX = {'apple': '/m/014j1m', 'banana': '/m/09qck', 'carrot': '/m/0fj52s', 'cucumber': '/m/015x4r',
       'lemon': '/m/09k_b', 'potato': '/m/05vtc', 'tomato': '/m/07j87'}
IMG = {'corn': '/m/053wn', 'garlic': '/m/0dbrl', 'onion': '/m/0dj75'}
ALL_IDS = set(BOX.values()) | set(IMG.values())
CAP = 120            # 클래스당 최대 장수
SEED = 0
OUT = '/content/indep_test'
WORK = '/content/oi'
os.makedirs(WORK, exist_ok=True)

def get(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path

rows = []
for split, bbox_name, img_name in [('validation', 'v5/validation-annotations-bbox.csv', 'v7/oidv7-val-annotations-human-imagelabels.csv'),
                                   ('test', 'v5/test-annotations-bbox.csv', 'v7/oidv7-test-annotations-human-imagelabels.csv')]:
    b = pd.read_csv(get(f'{G}/{bbox_name}', f'{WORK}/{split}-bbox.csv'))
    b = b[b.LabelName.isin(BOX.values())]
    b = b[(b.IsGroupOf == 0) & (b.IsDepiction == 0) & (b.IsTruncated == 0) & (b.IsOccluded == 0) & (b.IsInside == 0)]
    w, h = b.XMax - b.XMin, b.YMax - b.YMin
    b = b[(w * h >= 0.02) & (w >= 0.1) & (h >= 0.1)]
    inv = {v: k for k, v in BOX.items()}
    for r in b.itertuples():
        rows.append(dict(split=split, image=r.ImageID, label=inv[r.LabelName], kind='box',
                         x0=r.XMin, x1=r.XMax, y0=r.YMin, y1=r.YMax))
    il = pd.read_csv(get(f'{G}/{img_name}', f'{WORK}/{split}-imglabels.csv'))
    pos = il[(il.Confidence == 1) & il.LabelName.isin(ALL_IDS)]
    per_image = pos.groupby('ImageID').LabelName.apply(set)
    inv2 = {v: k for k, v in IMG.items()}
    for image, labs in per_image.items():
        if len(labs) == 1 and next(iter(labs)) in inv2:      # 대상 10종 중 이 라벨 하나만 붙은 이미지
            rows.append(dict(split=split, image=image, label=inv2[next(iter(labs))], kind='whole',
                             x0=0.0, x1=1.0, y0=0.0, y1=1.0))

df = pd.DataFrame(rows)
print('후보:', df.groupby(['label', 'kind']).size().to_dict())
rng = random.Random(SEED)
picked = []
for label, g in df.groupby('label'):
    g = g.sample(frac=1.0, random_state=SEED)
    g = g.groupby('image').head(2)           # 한 이미지에서 최대 2개
    picked.append(g.head(CAP))
sel = pd.concat(picked).reset_index(drop=True)
print('선택:', sel.groupby('label').size().to_dict())

if os.path.exists(OUT):
    shutil.rmtree(OUT)
cache = {}
def fetch(split, image):
    key = (split, image)
    if key not in cache:
        with urllib.request.urlopen(f'{S3}/{split}/{image}.jpg', timeout=60) as r:
            cache[key] = r.read()
    return cache[key]

def make(i):
    r = sel.iloc[i]
    try:
        im = Image.open(io.BytesIO(fetch(r.split, r.image))).convert('RGB')
        W, H = im.size
        if r.kind == 'box':
            pad = 0.08
            bw, bh = (r.x1 - r.x0) * W, (r.y1 - r.y0) * H
            box = (max(0, r.x0 * W - pad * bw), max(0, r.y0 * H - pad * bh),
                   min(W, r.x1 * W + pad * bw), min(H, r.y1 * H + pad * bh))
            im = im.crop(tuple(int(v) for v in box))
            if min(im.size) < 80:
                return i, 'small'
        im.thumbnail((640, 640))
        d = f'{OUT}/{r.label}'
        os.makedirs(d, exist_ok=True)
        im.save(f'{d}/{r.split}_{r.image}_{i}.jpg', quality=92)
        return i, 'ok'
    except Exception as e:
        return i, f'err {type(e).__name__}'

with cf.ThreadPoolExecutor(16) as ex:
    status = dict(ex.map(make, range(len(sel))))
sel['status'] = sel.index.map(status)
print('다운로드 결과:', sel.status.value_counts().to_dict())
ok = sel[sel.status == 'ok']
print('독립 테스트셋:', ok.groupby(['label', 'kind']).size().to_dict(), '| 합계', len(ok))
ok.to_csv('/content/indep_test_manifest.csv', index=False)

# 현재 모델(model_real.tflite) 평가: 보정 데이터는 원본 학습 데이터
data = '/content/data_indep'
os.makedirs(data, exist_ok=True)
for name, src in [('train', '/content/orig_train'), ('test', OUT)]:
    if not os.path.lexists(f'{data}/{name}'):
        os.symlink(src, f'{data}/{name}')
for mode in ('center_crop', 'squash'):
    out = f'/content/results_indep_{mode}'
    p = subprocess.run(['python', '/content/recipic_benchmark.py', '--model', '/content/model_real.tflite',
                        '--data', data, '--eval-split', 'test', '--preprocess', mode, '--runs', '10', '--out', out],
                       capture_output=True, text=True)
    print(f'--- 전처리 {mode} ---')
    print('\n'.join(l for l in p.stdout.splitlines() if '정확도' in l or '평가 이미지' in l))
    if p.returncode != 0:
        print(p.stderr[-1500:])
print(open('/content/results_indep_center_crop/benchmark.md', encoding='utf-8').read())

# 드라이브에 보관
D = '/content/drive/MyDrive/recipic_benchmark'
shutil.make_archive('/content/indep_test_v1', 'zip', OUT)
shutil.copy('/content/indep_test_v1.zip', f'{D}/indep_test_v1.zip')
shutil.copy('/content/indep_test_manifest.csv', f'{D}/indep_test_v1_manifest.csv')
for mode in ('center_crop', 'squash'):
    dst = f'{D}/results_indep_baseline_{mode}'
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(f'/content/results_indep_{mode}', dst, ignore=shutil.ignore_patterns('models'))
print('step02 완료')
