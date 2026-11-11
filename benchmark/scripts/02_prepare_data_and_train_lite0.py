# step08: 늘린 실제 사진(박스 클래스당 최대 600장)으로 EfficientNet-Lite0 재학습 (step06과 같은 학습 설정)
import os
D = '/content/drive/MyDrive/recipic_benchmark'
DATA_CODE = r'''
import os, io, csv, hashlib, random, shutil, subprocess, time, urllib.request, concurrent.futures as cf
import pandas as pd
from PIL import Image

G = 'https://storage.googleapis.com/openimages'
S3 = 'https://open-images-dataset.s3.amazonaws.com'
BOX = {'apple': '/m/014j1m', 'banana': '/m/09qck', 'carrot': '/m/0fj52s', 'cucumber': '/m/015x4r',
       'lemon': '/m/09k_b', 'potato': '/m/05vtc', 'tomato': '/m/07j87'}
IMG = {'corn': '/m/053wn', 'garlic': '/m/0dbrl', 'onion': '/m/0dj75'}
ALL = {**BOX, **IMG}
CAP = 600
SEED = 0
WORK = '/content/oi'
TRAIN = '/content/train_v3'          # train_v2/kaggle/<label>, train_v2/oi/<label>
os.makedirs(WORK, exist_ok=True)

# 1) Kaggle 원본에서 중복 제거, 짧은 변 256으로 줄여 저장
if os.path.exists(TRAIN):
    shutil.rmtree(TRAIN)
seen, kept = set(), 0
for label in sorted(ALL):
    src = f'/content/orig_train/{label}'
    dst = f'{TRAIN}/kaggle/{label}'
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        p = f'{src}/{name}'
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        try:
            im = Image.open(p).convert('RGB')
            s = 256 / min(im.size)
            if s < 1:
                im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
            im.save(f'{dst}/{h[:12]}.jpg', quality=92)
            kept += 1
        except Exception as e:
            print('skip', p, type(e).__name__)
print('Kaggle 중복 제거 후:', kept)

# 2) Open Images 학습 분할 주석을 내려받으며 대상 라벨만 걸러냄
def grep(url, ids, out):
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    pat = ' '.join(f"-e ',{i},'" for i in ids)
    t = time.time()
    subprocess.run(f"curl -s '{url}' | grep -F {pat} > {out}", shell=True, check=False)
    print(os.path.basename(out), os.path.getsize(out) // 1000, 'KB', round(time.time() - t), 's')

grep(f'{G}/v6/oidv6-train-annotations-bbox.csv', BOX.values(), f'{WORK}/train-bbox.csv')
grep(f'{G}/v7/oidv7-train-annotations-human-imagelabels.csv', ALL.values(), f'{WORK}/train-imglabels.csv')

cols = ['ImageID', 'Source', 'LabelName', 'Confidence', 'XMin', 'XMax', 'YMin', 'YMax',
        'IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside']
b = pd.read_csv(f'{WORK}/train-bbox.csv', header=None, usecols=range(13), names=cols)
b = b[(b.IsGroupOf == 0) & (b.IsDepiction == 0) & (b.IsInside == 0)]
w, h = b.XMax - b.XMin, b.YMax - b.YMin
b = b[(w * h >= 0.02) & (w >= 0.1) & (h >= 0.1)]
inv = {v: k for k, v in BOX.items()}
rows = [dict(image=r.ImageID, label=inv[r.LabelName], kind='box', x0=r.XMin, x1=r.XMax, y0=r.YMin, y1=r.YMax)
        for r in b.itertuples()]
il = pd.read_csv(f'{WORK}/train-imglabels.csv', header=None, names=['ImageID', 'Source', 'LabelName', 'Confidence'])
pos = il[il.Confidence == 1]
per_image = pos.groupby('ImageID').LabelName.apply(set)
inv2 = {v: k for k, v in IMG.items()}
for image, labs in per_image.items():
    if len(labs) == 1 and next(iter(labs)) in inv2:
        rows.append(dict(image=image, label=inv2[next(iter(labs))], kind='whole', x0=0.0, x1=1.0, y0=0.0, y1=1.0))
df = pd.DataFrame(rows)
print('Open Images 학습 후보:', df.groupby('label').size().to_dict())
picked = []
for label, g in df.groupby('label'):
    g = g.sample(frac=1.0, random_state=SEED).groupby('image').head(2)
    # 전체 이미지 라벨만 있는 클래스는 내려받을 수 없는 이미지가 많아 후보를 모두 시도한 뒤 CAP장만 남김
    picked.append(g if label in IMG else g.head(CAP))
sel = pd.concat(picked).reset_index(drop=True)

def make(i):
    r = sel.iloc[i]
    try:
        with urllib.request.urlopen(f'{S3}/train/{r.image}.jpg', timeout=60) as resp:
            im = Image.open(io.BytesIO(resp.read())).convert('RGB')
        W, H = im.size
        if r.kind == 'box':
            pad = 0.08
            bw, bh = (r.x1 - r.x0) * W, (r.y1 - r.y0) * H
            im = im.crop((int(max(0, r.x0 * W - pad * bw)), int(max(0, r.y0 * H - pad * bh)),
                          int(min(W, r.x1 * W + pad * bw)), int(min(H, r.y1 * H + pad * bh))))
            if min(im.size) < 80:
                return i, 'small'
        s = 256 / min(im.size)
        if s < 1:
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.BICUBIC)
        d = f'{TRAIN}/oi/{r.label}'
        os.makedirs(d, exist_ok=True)
        im.save(f'{d}/train_{r.image}_{i}.jpg', quality=92)
        return i, 'ok'
    except Exception as e:
        return i, f'err {type(e).__name__}'

t = time.time()
with cf.ThreadPoolExecutor(16) as ex:
    status = dict(ex.map(make, range(len(sel))))
sel['status'] = sel.index.map(status)
print('다운로드:', sel.status.value_counts().to_dict(), round(time.time() - t), 's')
print('Open Images 학습 데이터:', sel[sel.status == 'ok'].groupby('label').size().to_dict())
for label in IMG:
    d = f'{TRAIN}/oi/{label}'
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    for name in files[CAP:]:
        os.remove(f'{d}/{name}')
print('최종 Open Images 학습 데이터:', {l: len(os.listdir(f'{TRAIN}/oi/{l}')) for l in sorted(ALL) if os.path.isdir(f'{TRAIN}/oi/{l}')})
sel[sel.status == 'ok'].to_csv('/content/train_v3_oi_manifest.csv', index=False)
shutil.copy('/content/train_v3_oi_manifest.csv', f'{D}/train_v3_oi_manifest.csv')

# 평가 분할과 겹치는 이미지가 없는지 확인
test_ids = set(pd.read_csv('/content/indep_test_manifest.csv').image)
print('학습-테스트 이미지 ID 겹침:', len(test_ids & set(sel.image)))

'''
if not os.path.isdir('/content/train_v3/oi'):
    exec(DATA_CODE)

import os, sys, json, time, copy, random, zipfile, shutil, subprocess
import numpy as np, torch, torch.nn as nn, timm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

E = os.environ.get
TRAIN_ROOT = E('R_TRAIN', '/content/train_v3')
TEST_ROOT = E('R_TEST', '/content/indep_test')
OUT = E('R_OUT', '/content/v3b_out')
MODEL = E('R_MODEL', 'efficientnet_lite0')
DRIVE = E('R_DRIVE', '/content/drive/MyDrive/recipic_benchmark')
BENCH = E('R_BENCH', '/content/recipic_benchmark.py')
EPOCHS_B = int(E('R_EPOCHS_B', 15))
RES, BS, LR = int(E('R_RES', 224)), 32, 5e-4
PRETRAINED = E('R_PRETRAINED', '1') == '1'
SEED = 0
LABELS = ['apple', 'banana', 'carrot', 'corn', 'cucumber', 'garlic', 'lemon', 'onion', 'potato', 'tomato']
BOX_CLASSES = {'apple', 'banana', 'carrot', 'cucumber', 'lemon', 'potato', 'tomato'}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(os.cpu_count() or 2)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('장치:', DEV)
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

def items_of(root):
    out = []
    for i, label in enumerate(LABELS):
        d = os.path.join(root, label)
        if os.path.isdir(d):
            out += [(os.path.join(d, n), i) for n in sorted(os.listdir(d)) if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return out

def split(items, frac=0.15):
    rng = random.Random(SEED); tr, va = [], []
    for i in range(len(LABELS)):
        g = [x for x in items if x[1] == i]; rng.shuffle(g)
        k = max(1, round(len(g) * frac)) if len(g) > 1 else 0
        va += g[:k]; tr += g[k:]
    return tr, va

class DS(Dataset):
    def __init__(self, items, tf): self.items, self.tf = items, tf
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert('RGB')), y

BICUBIC = T.InterpolationMode.BICUBIC
train_tf = T.Compose([T.RandomResizedCrop(RES, scale=(0.5, 1.0), interpolation=BICUBIC), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.3, 0.3, 0.3, 0.03), T.RandomRotation(15), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize(RES + 32, interpolation=BICUBIC), T.CenterCrop(RES), T.ToTensor(), T.Normalize(MEAN, STD)])

@torch.no_grad()
def predict(net, items):
    net.eval(); preds = []
    for x, _ in DataLoader(DS(items, eval_tf), batch_size=64, num_workers=2):
        preds.append(net(x.to(DEV)).argmax(1).cpu())
    return torch.cat(preds).numpy() if preds else np.array([], dtype=int)

def report(net, items):
    y = np.array([t for _, t in items]); p = predict(net, items)
    per = {LABELS[i]: [int(((p == i) & (y == i)).sum()), int((y == i).sum())] for i in range(len(LABELS))}
    box = np.array([LABELS[t] in BOX_CLASSES for t in y])
    acc = lambda m: float((p[m] == y[m]).mean()) if m.any() else None
    return {'acc': float((p == y).mean()), 'acc_box7': acc(box), 'acc_whole3': acc(~box), 'n': int(len(y)), 'per_class': per}

def train(name, tr, va, epochs):
    print(f'\n=== {name}: 학습 {len(tr)}장 / 검증 {len(va)}장 / {epochs} epoch ===', flush=True)
    net = timm.create_model(MODEL, pretrained=PRETRAINED, num_classes=len(LABELS), drop_rate=0.2).to(DEV)
    counts = np.bincount([y for _, y in tr], minlength=len(LABELS)).astype(float)
    weights = [1.0 / counts[y] ** 0.5 for _, y in tr]          # 클래스 불균형 완화 (제곱근 역빈도)
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(tr), replacement=True)
    dl = DataLoader(DS(tr, train_tf), batch_size=BS, sampler=sampler, num_workers=2, drop_last=True)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=max(1, epochs * len(dl)), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        net.train(); t = time.time(); tot = n = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            loss = crit(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(y); n += len(y)
        yv = np.array([t_ for _, t_ in va]); acc = float((predict(net, va) == yv).mean()) if len(va) else 0.0
        hist.append({'epoch': ep + 1, 'loss': tot / max(1, n), 'val_acc': acc, 'sec': round(time.time() - t)})
        print(f'  epoch {ep + 1}/{epochs}  loss {tot / max(1, n):.3f}  val {acc * 100:.1f}%  {time.time() - t:.0f}s', flush=True)
        if acc > best:
            best, best_state = acc, copy.deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    net.to('cpu'); net.to(DEV)
    return net, {'best_val_acc': best, 'history': hist, 'train_n': len(tr), 'val_n': len(va)}

kag = items_of(os.path.join(TRAIN_ROOT, 'kaggle'))
oi = items_of(os.path.join(TRAIN_ROOT, 'oi'))
test = items_of(TEST_ROOT)
print('데이터: Kaggle', len(kag), '| Open Images 학습', len(oi), '| 독립 테스트', len(test))
k_tr, k_va = split(kag); o_tr, o_va = split(oi)

summary = {'labels': LABELS, 'res': RES, 'model': MODEL, 'runs': {}}
netB, infoB = train('C. Kaggle + Open Images(확대), ' + MODEL.split('.')[0], k_tr + o_tr, k_va + o_va, EPOCHS_B)
infoB['test'] = report(netB, test); summary['runs']['C_more_data_lite0'] = infoB
print('  C 독립 테스트:', json.dumps({k: v for k, v in infoB['test'].items() if k != 'per_class'}))
print('  C 클래스별:', infoB['test']['per_class'])
torch.save(netB.state_dict(), f'{OUT}/v3_model.pt')
json.dump(summary, open(f'{OUT}/train_summary.json', 'w'), ensure_ascii=False, indent=2)
if os.path.isdir(DRIVE):
    os.makedirs(f'{DRIVE}/v3b', exist_ok=True)
    for f in ('v3_model.pt', 'train_summary.json'):
        shutil.copy(f'{OUT}/{f}', f'{DRIVE}/v3b/{f}')

# ---- TFLite 변환 (기존 모델과 같은 입출력: NHWC, 0~1, softmax) ----
import litert_torch

class Wrapped(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))
    def forward(self, x):
        x = (x.permute(0, 3, 1, 2) - self.mean) / self.std
        return torch.softmax(self.net(x), dim=1)

tfl = f'{OUT}/recipic_lite0_v3.tflite'
DEV = 'cpu'
litert_torch.convert(Wrapped(netB.to('cpu')).eval(), (torch.rand(1, RES, RES, 3),)).export(tfl)
with zipfile.ZipFile(tfl, 'a') as z:
    z.writestr('labels.txt', '\n'.join(LABELS) + '\n')
print('TFLite 저장:', tfl, os.path.getsize(tfl) // 1000, 'KB')

# ---- 벤치마크: 독립 테스트셋에서 FP32 / INT8 ----
data = f'{OUT}/bench_data'; calib = f'{data}/train'
shutil.rmtree(data, ignore_errors=True)
for label in LABELS:
    os.makedirs(f'{calib}/{label}', exist_ok=True)
for p, y in k_tr + o_tr:
    os.symlink(p, f'{calib}/{LABELS[y]}/{os.path.basename(p)}')
os.symlink(TEST_ROOT, f'{data}/test')
res = f'{OUT}/results_indep_v3'
p = subprocess.run([sys.executable, BENCH, '--model', tfl, '--data', data, '--eval-split', 'test', '--out', res],
                   capture_output=True, text=True)
if p.returncode != 0:
    print(p.stdout[-800:], p.stderr[-1500:])
print(open(f'{res}/benchmark.md', encoding='utf-8').read())
if os.path.isdir(DRIVE):
    shutil.copy(tfl, f'{DRIVE}/v3b/recipic_lite0_v3.tflite')
    shutil.rmtree(f'{DRIVE}/v3b/results_indep_v3', ignore_errors=True)
    shutil.copytree(res, f'{DRIVE}/v3b/results_indep_v3')
print('step08 완료')
