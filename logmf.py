# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 14:20:21 2018

@author: lee
"""
import numpy as np
#import autograd.numpy as np
#from autograd import grad, jacobian, hessian
#from autograd.scipy.stats import norm
#from scipy.optimize import minimize
from scipy.misc import logsumexp
import re as myre
#from autograd.misc.flatten import flatten

#test
aadic={
        'A':0,
        'B':20,
        'C':1,
        'D':2,
        'E':3,
        
        'F':4,
        'G':5,
        'H':6,
        'I':7,
        'J':20,
        
        'K':8,
        'L':9,
        'M':10,
        'N':11,
        'O':20, 
        
        'P':12,
        'Q':13,
        'R':14,
        'S':15,
        'T':16,
        'U':20,
        
        'V':17,
        'W':18,
        'X':20,
        'Y':19,
        'Z':20,
        '-':20,
        '*':20,
        }
inverseaadic='ACDEFGHIKLMNPQRSTVWY-'
#def f(x,y,i):
#    return np.sum((2*x*y)*i+y*4)
#
#dx=grad(f)
#dy=grad(f,1)
#
#x=np.random.rand(4)
#y=np.random.rand(4)
#
#print(dx(x,y,6),dy(x,y,6))


def l2_norm(params):
    """Computes l2 norm of params by flattening them into a vector."""
    flattened= params.flatten()
    return np.dot(flattened, flattened)   

def softmax(x,ep=1e-6):
    #out[B,q]->[B,q]
    #print(np.sum(np.exp(x), axis=1,keepdims=True))
    
    return x - logsumexp(x, axis=-1, keepdims=True)

def cross_encropy_loss(out,y,weights):
    #print(out.shape,y.shape,weights.shape)
    # out and y ,weights shoud have same shape 
    return -1*np.sum((out)*y*weights)
def objfunu(u,ub,others):
    [v,vb,B,n,q,rank,msa,weights,rs]=others
    out=np.reshape(np.dot(u,v)+ub+vb,[B,n,q])
    out=softmax(out)
    return cross_encropy_loss(out,msa,weights)+rs[0]*l2_norm(u)+rs[1]*l2_norm(ub)+rs[2]*l2_norm(v)+rs[3]*l2_norm(vb)
def objfunv(v,vb,others): 
    [u,ub,B,n,q,rank,msa,weights,rs]=others
    out=np.reshape(np.dot(u,v)+ub+vb,[B,n,q])
    out=softmax(out)
    return cross_encropy_loss(out,msa,weights)+rs[0]*l2_norm(u)+rs[1]*l2_norm(ub)+rs[2]*l2_norm(v)+rs[3]*l2_norm(vb)
def infer(u,v,others):
    [B,n,q]=others
    out=np.reshape(np.dot(u,v),[B,n,q])
    out=softmax(out)
    return np.exp(out)
def npgradu(u,ub,others):
    [v,vb,B,n,q,rank,msa,weights,rs]=others
    out=np.reshape(np.dot(u,v)+ub+vb,[B,n,q])
    out=np.exp(softmax(out))    
    diffy=np.reshape((out-msa)*weights,[B,n*q])
    
    gu=np.dot(diffy,v.transpose())+2*rs[0]*u
    gub=np.sum(diffy, axis=1,keepdims=True)+2*rs[1]*ub
    return gu,gub
def npgradv(v,vb,others):
    [u,ub,B,n,q,rank,msa,weights,rs]=others
    out=np.reshape(np.dot(u,v)+ub+vb,[B,n,q])
    out=np.exp(softmax(out))
    diffy=np.reshape((out-msa)*weights,[B,n*q])
    gv=np.dot(u.transpose(),diffy)+2*rs[2]*v
    gvb=np.sum(diffy, axis=0,keepdims=True)+2*rs[3]*vb
    return gv,gvb  
def adam(gradu,gradv,initu,initv,initub,initvb, others,warmstart=False,model_=None,num_iters=500,step_size=0.01, b1=0.9, b2=0.999, eps=10**-8):
    mu=np.zeros_like(initu)
    mub=np.zeros_like(initub)
    vu,vub=np.zeros_like(initu),np.zeros_like(initub)
        
    mv=np.zeros_like(initv)
    mvb=np.zeros_like(initvb)
    vv,vvb=np.zeros_like(initv),np.zeros_like(initvb)
        
    [B,n,q,rank,msa,weights,rs]=others
    start=-1
    if warmstart:
        [mu,vu,mv,vv,mub,vub,mvb,vvb,start]=model_
    for i in range(start+1,num_iters):
        print(i,objfunu(initu,initub,[initv,initvb,B,n,q,rank,msa,weights,rs]))
        
        gu,gub=npgradu(initu,initub,[initv,initvb,B,n,q,rank,msa,weights,rs])
        #gu=gradu(initu,[initv,B,n,q,rank,msa,weights,rs])
        #print(npgu-gu)
        mu,mub=(1-b1)*gu+b1*mu,(1-b1)*gub+b1*mub
        vu,vub=(1-b2)*(gu**2) +b2*vu,(1-b2)*(gub**2) +b2*vub
        muhat,mubhat=mu/(1-b1**(i+1)),mub/(1-b1**(i+1))
        vuhat,vubhat=vu/(1- b2**(i + 1)) ,vub/(1- b2**(i + 1))        
        initu,initub = initu - step_size*muhat/(np.sqrt(vuhat) + eps) ,initub - step_size*mubhat/(np.sqrt(vubhat) + eps)

        gv,gvb=gradv(initv,initvb,[initu,initub,B,n,q,rank,msa,weights,rs])
        mv,mvb=(1-b1)*gv+b1*mv,(1-b1)*gvb+b1*mvb
        vv,vvb=(1-b2)*(gv**2) +b2*vv,(1-b2)*(gvb**2) +b2*vvb
        mvhat,mvbhat=mv/(1-b1**(i+1)),mvb/(1-b1**(i+1))
        vvhat,vvbhat=vv/(1- b2**(i + 1))  ,vvb/(1- b2**(i + 1))       
        initv ,initvb= initv - step_size*mvhat/(np.sqrt(vvhat) + eps)   ,initvb - step_size*mvbhat/(np.sqrt(vvbhat) + eps)       
    return [initu,initub,initv,initvb],[mu,vu,mv,vv,mub,vub,mvb,vvb,start]            
def optmize(rank,msa,weights,rs,t=100):
    [B,n,q]=msa.shape
    u=np.random.normal(0.,1.0/np.sqrt(rank),[B,rank])
    ub=np.random.normal(0.,1.0/np.sqrt(rank),[B,1])
    v=np.random.normal(0.,1.0/np.sqrt(rank),[rank,n*q])
    vb=np.random.normal(0.,1.0/np.sqrt(rank),[1,n*q])
    others=[B,n,q,rank,msa,weights,rs]
    [u,ub,v,vb],model=adam(npgradu,npgradv,u,v,ub,vb, others,num_iters=t)
    return [u,ub,v,vb],model
def read_msa(file_path,aadic):    
    lines=open(file_path).readlines()  
    lines=[line.strip() for line in lines]   
    n=len(lines)
    d=len(lines[0]) #CR AND LF  
    mask=np.ones([n,d],dtype=int)
    msa=np.zeros([n,d],dtype=int)
    for i in  range(n):
        aline=lines[i]
        tmpaline=myre.sub('[A-Z]',' ',aline)
        if tmpaline[0]=='-':
            mask[i,:len(tmpaline.split()[0])]=0
        if tmpaline[-1]=='-':
            mask[i,-len(tmpaline.split()[-1]):]=0
        for j in range(d):
            msa[i,j]=aadic[aline[j]]
    return msa,mask
    
    
def runonetarget(target):
    msafile='multi_test/'+target+'.aln'
    msa_,mask=read_msa(msafile,aadic)
    
    msa=np.eye(21)[msa_]
    B,n,q=msa.shape
    [u,ub,v,vb],model=optmize(5523,msa,mask.reshape([B,n,1]),[1,1,10,10],t=100)
    out=np.reshape(np.dot(u,v)+ub+vb,[B,n,q])
    out=np.exp(softmax(out))
    out=np.argmax(out,axis=-1)
    out=msa_*mask+(1-mask)*out
    
    lines=[]
    for i in range(B):
        aline=[]
        for j in range(n):
            aline.append(inverseaadic[out[i,j]])
        lines.append(''.join(aline))
    wfile=open('pseudoaln/'+target+'.aln','w')
    for aline in lines:
        wfile.write(aline+'\n')
    wfile.close()
    
if __name__ == "__main__":
    

    target='T0783'

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    