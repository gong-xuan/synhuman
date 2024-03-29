"""
Parts of the code are adapted from https://github.com/akanazawa/hmr
"""

import numpy as np


class AverageMeter():
    def __init__(self):
        self.count = 0
        self.sum = 0
        
    def update(self, value, n=1):
        self.count += n
        self.sum += value*n
        # print(self.count)

    def average(self):
        return self.sum/(self.count)

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def procrustes_analysis_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat


def scale_and_translation_transform_batch(P, T):
    """
    First Normalises batch of input 3D meshes P such that each mesh has mean (0, 0, 0) and
    RMS distance from mean = 1.
    Then transforms P such that it has the same mean and RMSD as T.
    :param P: (batch_size, N, 3) batch of N 3D meshes to transform.
    :param T: (batch_size, N, 3) batch of N reference 3D meshes.
    :return: P transformed
    """
    P_mean = np.mean(P, axis=1, keepdims=True)
    P_trans = P - P_mean
    P_scale = np.sqrt(np.sum(P_trans ** 2, axis=(1, 2), keepdims=True) / P.shape[1])
    P_normalised = P_trans / P_scale

    T_mean = np.mean(T, axis=1, keepdims=True)
    T_scale = np.sqrt(np.sum((T - T_mean) ** 2, axis=(1, 2), keepdims=True) / T.shape[1])

    P_transformed = P_normalised * T_scale + T_mean

    return P_transformed

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat

def compute_similarity_transform_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat

def reconstruction_error(S1, S2, reduction='mean'):
    """Do Procrustes alignment and compute reconstruction error."""
    S1_hat = compute_similarity_transform_batch(S1, S2)
    re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1)).mean(axis=-1)
    if reduction == 'mean':
        re = re.mean()
    elif reduction == 'sum':
        re = re.sum()
    elif reduction == 'pck':
        re = np.sqrt( ((S1_hat - S2)** 2).sum(axis=-1))*1000#.mean(axis=-1)
        re = (re<=150).astype(float).sum(axis=-1)/17
    return re


def cal_pve(pred_vertices, target_vertices):
    pve_batch = np.linalg.norm(pred_vertices - target_vertices,
                                       axis=-1)  # (bsize, 6890)
    return np.mean(pve_batch)

def cal_pve_pa(pred_vertices, target_vertices):
    pred_vertices_pa = procrustes_analysis_batch(pred_vertices, target_vertices)
    pve_pa_batch = np.linalg.norm(pred_vertices_pa - target_vertices,
                                        axis=-1)  # (bsize, 6890)
    return np.mean(pve_pa_batch)

def cal_pve_ts_sc(pred_reposed_vertices, target_reposed_vertices):
    pred_reposed_vertices_sc = scale_and_translation_transform_batch(pred_reposed_vertices,
                                                                    target_reposed_vertices)
    # pred_reposed_vertices_sc = pred_reposed_vertices                                                  
    pvet_sc_batch = np.linalg.norm(pred_reposed_vertices_sc - target_reposed_vertices,
                                axis=-1)  # (bs, 6890)
    return np.mean(pvet_sc_batch)

def cal_mpjpe_pa(pred_joints3D_h36mlsp, target_joints3D_h36mlsp):
    pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
    
    # pred_joints3D_h36mlsp_pa = pred_joints3D_h36mlsp
    # print(pred_joints3D_h36mlsp_pa)
    # print(target_joints3D_h36mlsp)
    # print(pred_joints3D_h36mlsp_pa.shape)
    # print(target_joints3D_h36mlsp.shape)
    # import ipdb; ipdb.set_trace()
    
    mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp,
                                axis=-1)  # (bsize, 14)
    # import ipdb; ipdb.set_trace()
    
    mpjpe_pa_batch = np.mean(mpjpe_pa_batch)
    # print(mpjpe_pa_batch)
    return mpjpe_pa_batch

def cal_mpjpe(pred_joints3D_h36mlsp, target_joints3D_h36mlsp):
    
    mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp- target_joints3D_h36mlsp,
                                axis=-1)  # (bsize, 14)
    # import ipdb; ipdb.set_trace()
    
    mpjpe_pa_batch = np.mean(mpjpe_pa_batch)
    # print(mpjpe_pa_batch)
    return mpjpe_pa_batch

def cal_mpjpe_sc(pred_joints3D_h36mlsp, target_joints3D_h36mlsp):
    pred_joints3D_h36mlsp_sc = scale_and_translation_transform_batch(pred_joints3D_h36mlsp,
                                                                             target_joints3D_h36mlsp)
    mpjpe_pa_batch = np.linalg.norm(pred_joints3D_h36mlsp_sc - target_joints3D_h36mlsp,
                                axis=-1)  # (bsize, 14)
    # import ipdb; ipdb.set_trace()
    
    mpjpe_pa_batch = np.mean(mpjpe_pa_batch)
    # print(mpjpe_pa_batch)
    return mpjpe_pa_batch

def cal_pck_pa(pred_joints3D_h36mlsp, target_joints3D_h36mlsp):
    pred_joints3D_h36mlsp_pa = procrustes_analysis_batch(pred_joints3D_h36mlsp,
                                                                 target_joints3D_h36mlsp)
    
    error = np.sqrt( ((pred_joints3D_h36mlsp_pa - target_joints3D_h36mlsp)** 2).sum(axis=-1))*1000#.mean(axis=-1)
    pck_batch = (error<=150).astype(float).sum(axis=-1)/error.shape[-1]
    thresh=[0+n*5 for n in range(1, 31)]
    batch_size = pred_joints3D_h36mlsp.shape[0]
    pck_curve = np.zeros((30, batch_size))
    for n,t in enumerate(thresh):
        pck_curve[n] = (error<=t).astype(float).sum(axis=-1)/error.shape[-1]
    # import ipdb; ipdb.set_trace()
    return np.mean(pck_batch), np.mean(pck_curve)