"""DEST-CB1: image-only destination ownership prototype. No simulator identity."""
import numpy as np

def prompt_box(masks, point):
    masks=np.asarray(masks,bool)
    if masks.ndim==2:masks=masks[None]
    x,y=np.rint(point).astype(int)
    if not len(masks) or not (0<=x<masks.shape[2] and 0<=y<masks.shape[1]):return None
    choices=[i for i,m in enumerate(masks) if m[y,x] and m.any()]
    if not choices:return None
    i=max(choices,key=lambda j:(int(masks[j].sum()),-j))
    ys,xs=np.where(masks[i])
    return np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1],dtype=np.float32)

def segment_current(model,processor,image,box):
    if box is None:return np.zeros((image.height,image.width),bool),None
    state=processor.set_image(image)
    masks,scores,_=model.predict_inst(state,box=np.asarray(box,np.float32),multimask_output=False)
    masks=np.asarray(masks,bool)
    if masks.ndim==2:masks=masks[None]
    if len(masks)!=1:raise ValueError('Expected one box-prompt mask')
    if masks[0].shape!=(image.height,image.width):raise ValueError('Unexpected SAM mask dimensions')
    return masks[0],np.asarray(scores).reshape(-1).tolist()

def replace_destination_claim(world,floating,claimed_without_destination,w2p,current_mask):
    world=np.asarray(world);h,w=current_mask.shape
    projected=np.c_[world,np.ones(len(world))]@np.asarray(w2p).T
    xy=np.rint(projected[:,:2]/projected[:,2:3]).astype(int)
    inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    destination=np.zeros(len(world),bool)
    ids=np.flatnonzero(inside);destination[ids]=current_mask[xy[ids,1],xy[ids,0]]
    other=np.asarray(floating,bool)&~(np.asarray(claimed_without_destination,bool)|destination)
    return world[other],world[destination],destination,other
