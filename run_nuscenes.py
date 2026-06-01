#!/usr/bin/env python3
"""Re-run ONLY the nuScenes cross-sensor transfer, reusing the KITTI checkpoints
(data/runs/kitti_omumae_full/<variant>/ckpt.pt) + the already-built DINOv2 cache.
Fixes the 'Too many open files' (Errno 24) FD leak that crashed the in-notebook sweep."""
import os, json, resource
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.auto import tqdm

# ---------------- FD-leak fix ----------------
torch.multiprocessing.set_sharing_strategy('file_system')   # avoid FD-based tensor sharing
try:
    _s, _h = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(1048576, _h), _h))   # raise open-file ceiling
    print('RLIMIT_NOFILE ->', resource.getrlimit(resource.RLIMIT_NOFILE))
except Exception as e:
    print('ulimit raise failed (non-fatal):', e)

SEED = 42; np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); IS_CUDA = device.type == 'cuda'
print('device:', device, torch.cuda.get_device_name(0) if IS_CUDA else '')

ROOT = Path('/workspace/full_omu_mae')
OUT_BASE = ROOT / 'data' / 'runs' / 'kitti_omumae_full'
NUSC_ROOT = ROOT / 'data' / 'nuscenes'
NUSC_VERSION = 'v1.0-trainval'
GRID = (128, 128, 32); VOX_RES = (0.4, 0.4, 0.4); H_DIM = 256; IMG_SIZE = 224
NUSC_SEEDS = [0, 1, 2]          # set to [0] to cut runtime ~3x
PROBE_EPOCHS = 5; PROBE_LR = 5e-3
label_fractions = [0.01, 0.05, 0.10, 1.00]
FEAT_DIM = 64; DINOV2_MODEL = 'dinov2_vitb14'; SLIDR_PROJ_DIM = 64

# ---------------- models (verbatim from the notebook) ----------------
def conv_bn_act(i, o, k=3, s=1, p=1):
    return nn.Sequential(nn.Conv3d(i, o, k, stride=s, padding=p, bias=False), nn.GroupNorm(8, o), nn.GELU())
def upsample_conv(i, o):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
                         nn.Conv3d(i, o, 3, padding=1, bias=False), nn.GroupNorm(8, o), nn.GELU())
class OMUMAE(nn.Module):
    def __init__(self, image_feat_dim=384, hidden=64):
        super().__init__(); in_ch = 1 + image_feat_dim + 1
        self.in_proj = nn.Conv3d(in_ch, hidden, 1)
        self.enc = nn.Sequential(conv_bn_act(hidden,hidden),conv_bn_act(hidden,hidden*2,s=2),conv_bn_act(hidden*2,hidden*2),conv_bn_act(hidden*2,hidden*4,s=2),conv_bn_act(hidden*4,hidden*4))
        self.dec = nn.Sequential(conv_bn_act(hidden*4,hidden*4),upsample_conv(hidden*4,hidden*2),conv_bn_act(hidden*2,hidden*2),upsample_conv(hidden*2,hidden),conv_bn_act(hidden,hidden))
        self.head_occ = nn.Conv3d(hidden,1,1); self.head_feat = nn.Conv3d(hidden,image_feat_dim,1)
    def encode(self, occ, feat, mask=None):
        if mask is not None: occ_in=occ*(1-mask); feat_in=feat*(1-mask); mind=mask
        else: occ_in,feat_in=occ,feat; mind=torch.zeros_like(occ)
        return self.enc(self.in_proj(torch.cat([occ_in,feat_in,mind],1)))
class OccupancyMAEBaseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.in_proj = nn.Conv3d(2, hidden, 1)
        self.enc = nn.Sequential(conv_bn_act(hidden,hidden),conv_bn_act(hidden,hidden*2,s=2),conv_bn_act(hidden*2,hidden*2),conv_bn_act(hidden*2,hidden*4,s=2),conv_bn_act(hidden*4,hidden*4))
        self.dec = nn.Sequential(conv_bn_act(hidden*4,hidden*4),upsample_conv(hidden*4,hidden*2),conv_bn_act(hidden*2,hidden*2),upsample_conv(hidden*2,hidden),conv_bn_act(hidden,hidden))
        self.head_occ = nn.Conv3d(hidden,1,1)
    def encode(self, occ, feat=None, mask=None):
        mind = mask if mask is not None else torch.zeros_like(occ)
        occ_in = occ*(1-mask) if mask is not None else occ
        return self.enc(self.in_proj(torch.cat([occ_in,mind],1)))
class SLidRVoxel(nn.Module):
    def __init__(self, hidden=64, proj_dim=64, image_feat_dim=384):
        super().__init__(); self.in_proj=nn.Conv3d(1,hidden,1)
        self.enc=nn.Sequential(conv_bn_act(hidden,hidden),conv_bn_act(hidden,hidden*2,s=2),conv_bn_act(hidden*2,hidden*2),conv_bn_act(hidden*2,hidden*4,s=2),conv_bn_act(hidden*4,hidden*4))
        self.dec=nn.Sequential(conv_bn_act(hidden*4,hidden*4),upsample_conv(hidden*4,hidden*2),conv_bn_act(hidden*2,hidden*2),upsample_conv(hidden*2,hidden),conv_bn_act(hidden,hidden))
        self.proj3d=nn.Conv3d(hidden,proj_dim,1); self.proj_img=nn.Linear(image_feat_dim,proj_dim)
    def encode(self, occ, feat=None, mask=None): return self.enc(self.in_proj(occ))
class FrozenImageEncoder(nn.Module):
    def __init__(self, model_name='dinov2_vitb14'):
        super().__init__()
        self.embed_dim={'dinov2_vits14':384,'dinov2_vitb14':768,'dinov2_vitl14':1024,'dinov2_vitg14':1536}.get(model_name,384)
        self.patch_grid=16
        self.backbone=torch.hub.load('facebookresearch/dinov2', model_name, pretrained=True, trust_repo=True)
        for p in self.backbone.parameters(): p.requires_grad_(False)
        self.backbone.eval()
    @torch.no_grad()
    def forward(self, image):
        out=self.backbone.forward_features(image)
        return out['x_norm_patchtokens'] if isinstance(out,dict) else out
class ProbeHead(nn.Module):
    def __init__(self, in_dim, num_classes, target_grid):
        super().__init__(); self.classifier=nn.Conv3d(in_dim,num_classes,1); self.target_grid=target_grid
    def forward(self, h): return F.interpolate(self.classifier(h), size=self.target_grid, mode='trilinear', align_corners=False)

print('loading DINOv2', DINOV2_MODEL, '...')
img_encoder = FrozenImageEncoder(DINOV2_MODEL).to(device)
VARIANTS = {
 'random': dict(pretrain=False, model_class='OMUMAE',              use_dinov2_input=True),
 'occmae': dict(pretrain=True,  model_class='OccupancyMAEBaseline', use_dinov2_input=False),
 'nomask': dict(pretrain=True,  model_class='OMUMAE',              use_dinov2_input=True),
 'full':   dict(pretrain=True,  model_class='OMUMAE',              use_dinov2_input=True),
 'slidr':  dict(pretrain=True,  model_class='SLidRVoxel',          use_dinov2_input=False),
}
def make_model(v):
    c=VARIANTS[v]['model_class']
    if c=='OMUMAE': return OMUMAE(image_feat_dim=img_encoder.embed_dim, hidden=FEAT_DIM).to(device)
    if c=='OccupancyMAEBaseline': return OccupancyMAEBaseline(hidden=FEAT_DIM).to(device)
    if c=='SLidRVoxel': return SLidRVoxel(hidden=FEAT_DIM, proj_dim=SLIDR_PROJ_DIM, image_feat_dim=img_encoder.embed_dim).to(device)
    raise ValueError(c)

# ---------------- voxelization ----------------
def voxelize_lidar_only(points, grid_size, voxel_size):
    B,N=points.shape[0],points.shape[1]; X,Y,Z=grid_size; vx,vy,vz=voxel_size; ox=X*vx/2;oy=Y*vy/2;oz=Z*vz/2
    pts=points[...,:3]
    ix=((pts[...,0]+ox)/vx).long().clamp(0,X-1); iy=((pts[...,1]+oy)/vy).long().clamp(0,Y-1); iz=((pts[...,2]+oz)/vz).long().clamp(0,Z-1)
    occ=torch.zeros(B,1,X,Y,Z,device=points.device); b=torch.arange(B,device=points.device).unsqueeze(-1).expand(-1,N)
    occ[b,0,ix,iy,iz]=1.0; return occ
def voxelize_with_feats_multicam(points, dino_feats_multi, P2_stack, Tr_stack, grid_size, voxel_size, image_size, patch_grid):
    B,N=points.shape[0],points.shape[1]; K=dino_feats_multi.shape[1]; X,Y,Z=grid_size; vx,vy,vz=voxel_size
    ox=X*vx/2;oy=Y*vy/2;oz=Z*vz/2; D=dino_feats_multi.shape[-1]; pts=points[...,:3]
    ix=((pts[...,0]+ox)/vx).long().clamp(0,X-1); iy=((pts[...,1]+oy)/vy).long().clamp(0,Y-1); iz=((pts[...,2]+oz)/vz).long().clamp(0,Z-1)
    occ=torch.zeros(B,1,X,Y,Z,device=points.device); feat=torch.zeros(B,D,X,Y,Z,device=points.device); cnt=torch.zeros(B,1,X,Y,Z,device=points.device)
    bi=torch.arange(B,device=points.device).unsqueeze(-1).expand(-1,N); occ[bi,0,ix,iy,iz]=1.0
    pts_h=torch.cat([pts,torch.ones(B,N,1,device=pts.device)],-1)
    for c in range(K):
        cam=(Tr_stack[:,c]@pts_h.transpose(1,2)).transpose(1,2); cam_h=torch.cat([cam,torch.ones(B,N,1,device=pts.device)],-1)
        img_xy=(P2_stack[:,c]@cam_h.transpose(1,2)).transpose(1,2); in_front=img_xy[...,2]>0.1
        u=(img_xy[...,0]/img_xy[...,2].clamp(min=1e-3)).long(); v=(img_xy[...,1]/img_xy[...,2].clamp(min=1e-3)).long()
        valid=in_front&(u>=0)&(u<image_size)&(v>=0)&(v<image_size)
        if not valid.any(): continue
        pu=(u.float()/image_size*patch_grid).long().clamp(0,patch_grid-1); pv=(v.float()/image_size*patch_grid).long().clamp(0,patch_grid-1)
        patch_idx=(pv*patch_grid+pu).clamp(0,patch_grid*patch_grid-1)
        f_pt=torch.gather(dino_feats_multi[:,c],1,patch_idx.unsqueeze(-1).expand(-1,-1,D))*valid.unsqueeze(-1).float()
        for b in range(B):
            if valid[b].sum()==0: continue
            idx_v=(ix[b]*Y*Z+iy[b]*Z+iz[b])
            ff=feat[b].permute(1,2,3,0).reshape(-1,D); fc=cnt[b].permute(1,2,3,0).reshape(-1,1)
            ff.index_add_(0,idx_v,f_pt[b]); fc.index_add_(0,idx_v,valid[b].unsqueeze(-1).float())
            feat[b]=ff.reshape(X,Y,Z,D).permute(3,0,1,2); cnt[b]=fc.reshape(X,Y,Z,1).permute(3,0,1,2)
    return occ, feat/cnt.clamp(min=1.0)
def voxelize_labels_np(points, raw_labels, grid_size, voxel_size, remap_lut, n_classes):
    X,Y,Z=grid_size; vx,vy,vz=voxel_size; ox=X*vx/2;oy=Y*vy/2;oz=Z*vz/2
    sem=remap_lut[raw_labels.astype(np.int64)]; valid=sem>0
    if not valid.any(): return np.zeros((X,Y,Z),np.int32)
    ix=np.clip(((points[:,0]+ox)/vx).astype(np.int32),0,X-1); iy=np.clip(((points[:,1]+oy)/vy).astype(np.int32),0,Y-1); iz=np.clip(((points[:,2]+oz)/vz).astype(np.int32),0,Z-1)
    counts=np.zeros((X,Y,Z,n_classes),np.int32); np.add.at(counts,(ix[valid],iy[valid],iz[valid],sem[valid]),1)
    has=counts.sum(-1)>0; out=np.zeros((X,Y,Z),np.int32); out[has]=counts[has].argmax(-1); return out

# ---------------- nuScenes calib + class map ----------------
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as _ns
from pyquaternion import Quaternion
def _inv(R,t): Ri=R.T; return Ri,-Ri@t
def _rt(R,t): M=np.eye(4,dtype=np.float32); M[:3,:3]=R; M[:3,3]=t; return M[:3,:].astype(np.float32)
def _comp(A,B): A4=np.eye(4,dtype=np.float32);A4[:3,:]=A; B4=np.eye(4,dtype=np.float32);B4[:3,:]=B; return (A4@B4)[:3,:].astype(np.float32)
NUSC_CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_BACK_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_FRONT_LEFT']
def build_nusc_calib_multicam(nusc, tok, target_image_size=224, cameras=NUSC_CAMERAS):
    s=nusc.get('sample',tok); ld=nusc.get('sample_data',s['data']['LIDAR_TOP']); cl=nusc.get('calibrated_sensor',ld['calibrated_sensor_token'])
    Tegolid=_rt(Quaternion(cl['rotation']).rotation_matrix.astype(np.float32), np.asarray(cl['translation'],np.float32))
    K=len(cameras); Tr=np.zeros((K,3,4),np.float32); P2=np.zeros((K,3,4),np.float32); paths=[]
    for ci,cam in enumerate(cameras):
        cd=nusc.get('sample_data',s['data'][cam]); cs=nusc.get('calibrated_sensor',cd['calibrated_sensor_token'])
        Rc=Quaternion(cs['rotation']).rotation_matrix.astype(np.float32); tc=np.asarray(cs['translation'],np.float32)
        Ri,ti=_inv(Rc,tc); Tr[ci]=_comp(_rt(Ri,ti),Tegolid)
        Km=np.asarray(cs['camera_intrinsic'],np.float32); sx=target_image_size/cd['width']; sy=target_image_size/cd['height']
        P2[ci]=np.hstack([np.array([[sx,0,0],[0,sy,0],[0,0,1]],np.float32)@Km, np.zeros((3,1),np.float32)])
        paths.append(os.path.join(nusc.dataroot, cd['filename']))
    return {'Tr_stack':Tr,'P2_stack':P2,'cam_paths':paths}
NUSC_RAW_TO_CHALLENGE={0:0,1:0,2:7,3:7,4:7,5:0,6:7,7:0,8:0,9:1,10:0,11:0,12:8,13:0,14:2,15:3,16:3,17:4,18:5,19:0,20:0,21:6,22:9,23:10,24:11,25:12,26:13,27:14,28:15,29:0,30:16,31:0}
NUM_CLASSES_NUSC=17; _REMAP_LUT=np.zeros(32,np.int64)
for _r,_c in NUSC_RAW_TO_CHALLENGE.items(): _REMAP_LUT[_r]=_c

nusc = NuScenes(version=NUSC_VERSION, dataroot=str(NUSC_ROOT), verbose=False)
_tr,_va=set(_ns.train),set(_ns.val)
def _sn(t): s=nusc.get('sample',t); return nusc.get('scene',s['scene_token'])['name']
_all=[s['token'] for s in nusc.sample]
nusc_train_tokens=[t for t in _all if _sn(t) in _tr]; nusc_val_tokens=[t for t in _all if _sn(t) in _va]
print(f'official split: train={len(nusc_train_tokens)} val={len(nusc_val_tokens)}')
DINO_CACHE = OUT_BASE / f'dino_cache_{NUSC_VERSION}'; DINO_CACHE.mkdir(parents=True, exist_ok=True)

# ---------------- DINOv2 cache (reuses existing) ----------------
_MEAN=np.array([0.485,0.456,0.406],np.float32); _STD=np.array([0.229,0.224,0.225],np.float32)
def _img224(p):
    im=Image.open(p).convert('RGB').resize((224,224)); a=np.asarray(im,np.float32)/255.0
    return np.transpose((a-_MEAN)/_STD,(2,0,1))
@torch.no_grad()
def precompute(tokens):
    todo=[t for t in tokens if not (DINO_CACHE/f'{t}.npy').exists()]
    print(f'dino cache: {len(tokens)-len(todo)}/{len(tokens)} present; computing {len(todo)}')
    for t in tqdm(todo, desc='dino-cache'):
        cal=build_nusc_calib_multicam(nusc,t); imgs=np.stack([_img224(p) for p in cal['cam_paths']])
        tok=img_encoder(torch.from_numpy(imgs).to(device)); np.save(DINO_CACHE/f'{t}.npy', tok.float().cpu().numpy().astype(np.float16))
precompute(nusc_train_tokens+nusc_val_tokens)

class NuScenesCached(Dataset):
    def __init__(self, tokens, n_points=16384): self.tokens=list(tokens); self.n=n_points
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i):
        t=self.tokens[i]; cal=build_nusc_calib_multicam(nusc,t)
        s=nusc.get('sample',t); ld=nusc.get('sample_data',s['data']['LIDAR_TOP'])
        lp=os.path.join(nusc.dataroot,ld['filename']); sp=os.path.join(nusc.dataroot,'lidarseg',NUSC_VERSION,ld['token']+'_lidarseg.bin')
        pts=np.fromfile(lp,np.float32).reshape(-1,5)[:,:4]; lbl=np.fromfile(sp,np.uint8)
        if len(lbl)!=pts.shape[0]: n=min(len(lbl),pts.shape[0]); pts=pts[:n]; lbl=lbl[:n]
        vox=voxelize_labels_np(pts,lbl,GRID,VOX_RES,_REMAP_LUT,NUM_CLASSES_NUSC)
        N=pts.shape[0]; pts=pts[np.random.choice(N,self.n,replace=False)] if N>=self.n else np.concatenate([pts,np.zeros((self.n-N,4),np.float32)])
        dino=np.load(DINO_CACHE/f'{t}.npy').astype(np.float32)
        return {'dino':torch.from_numpy(dino),'points':torch.from_numpy(pts),'P2':torch.from_numpy(cal['P2_stack']),'Tr':torch.from_numpy(cal['Tr_stack']),'vox_lbl':torch.from_numpy(vox).long()}
def collate(b): return {k:torch.stack([x[k] for x in b]) for k in b[0]}
@torch.no_grad()
def enc_feats(model,dino,points,P2,Tr,use_dino):
    if use_dino:
        occ,feat=voxelize_with_feats_multicam(points,dino,P2,Tr,GRID,VOX_RES,IMG_SIZE,img_encoder.patch_grid)
        return model.encode(occ,feat,mask=None)
    return model.encode(voxelize_lidar_only(points,GRID,VOX_RES),mask=None)

# ---------------- FIXED sweep (no FD leak) ----------------
def loader(ds, shuffle):
    return DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=4,
                      pin_memory=False, persistent_workers=False, collate_fn=collate)   # <-- the fix
def train_eval(model, use_dino, tr_ds, va_ds, frac, seed):
    g=np.random.RandomState(seed)
    tr=Subset(tr_ds, g.choice(len(tr_ds),max(1,int(len(tr_ds)*frac)),replace=False).tolist()) if frac<1.0 else tr_ds
    torch.manual_seed(seed); head=ProbeHead(H_DIM,NUM_CLASSES_NUSC,GRID).to(device)
    opt=torch.optim.AdamW(head.parameters(),lr=PROBE_LR,weight_decay=1e-4); model.eval(); head.train()
    for ep in range(PROBE_EPOCHS):
        ld=loader(tr,True)
        for batch in tqdm(ld, desc=f's{seed} f{int(frac*100)}% ep{ep+1}', leave=False):
            h=enc_feats(model,batch['dino'].to(device),batch['points'].to(device),batch['P2'].to(device),batch['Tr'].to(device),use_dino)
            loss=F.cross_entropy(head(h),batch['vox_lbl'].to(device),ignore_index=0)
            opt.zero_grad(); loss.backward(); opt.step()
        del ld
    head.eval(); inter=np.zeros(NUM_CLASSES_NUSC,np.int64); union=np.zeros(NUM_CLASSES_NUSC,np.int64)
    with torch.no_grad():
        ld=loader(va_ds,False)
        for batch in tqdm(ld, desc=f's{seed} eval', leave=False):
            pred=head(enc_feats(model,batch['dino'].to(device),batch['points'].to(device),batch['P2'].to(device),batch['Tr'].to(device),use_dino)).argmax(1)
            lbl=batch['vox_lbl'].to(device)
            for c in range(1,NUM_CLASSES_NUSC):
                p=(pred==c); gt=(lbl==c); inter[c]+=int((p&gt).sum()); union[c]+=int((p|gt).sum())
        del ld
    valid=union>0; return float((inter[valid]/np.maximum(union[valid],1)).mean())

tr_ds=NuScenesCached(nusc_train_tokens); va_ds=NuScenesCached(nusc_val_tokens)
raw={v:{f:[] for f in label_fractions} for v in VARIANTS}
for seed in NUSC_SEEDS:
    for variant in VARIANTS:
        vcfg=VARIANTS[variant]; model=make_model(variant)
        if vcfg['pretrain']:
            ck=torch.load(OUT_BASE/variant/'ckpt.pt',map_location=device,weights_only=False); model.load_state_dict(ck['model'])
        ud=vcfg['use_dinov2_input']
        for frac in label_fractions:
            m=train_eval(model,ud,tr_ds,va_ds,frac,seed); raw[variant][frac].append(m)
            print(f'  nusc seed{seed} {variant} @ {int(frac*100)}%: {m*100:.2f}', flush=True)
        del model
        if IS_CUDA: torch.cuda.empty_cache()
agg={v:{f:{'mean':float(np.mean(raw[v][f])*100),'std':float(np.std(raw[v][f])*100)} for f in label_fractions} for v in VARIANTS}
with open(OUT_BASE/'nuscenes_transfer_results.json','w') as fh:
    json.dump({'version':NUSC_VERSION,'seeds':NUSC_SEEDS,'n_val':len(nusc_val_tokens),
               'agg':{v:{str(f):agg[v][f] for f in label_fractions} for v in VARIANTS}}, fh, indent=2)
print('\n=== nuScenes transfer (official val, mean+/-std), mIoU % ===')
print(f'{"label%":>7s}'+''.join(f'  {v:>13s}' for v in VARIANTS))
for f in label_fractions:
    print(f'{int(f*100):>6d}%'+''.join(f'  {agg[v][f]["mean"]:>6.2f}+/-{agg[v][f]["std"]:<4.2f}' for v in VARIANTS))
# collect into results/
import shutil; res=ROOT/'results'; res.mkdir(exist_ok=True); shutil.copy(OUT_BASE/'nuscenes_transfer_results.json', res/'nuscenes_transfer_results.json')
print('\n=== nuScenes DONE -> results/nuscenes_transfer_results.json ===')
