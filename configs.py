LOG_DIR = '../exp_out/cra/logs'
CKPT_DIR =  '../exp_out/cra/ckpts'
ROOT_PATH = '/a2il/data/xuangong/3dhmr'
SMPL_MODEL_DIR = f'{ROOT_PATH}/additional/smpl'
SMPL_FACES_PATH = f'{ROOT_PATH}/additional/smpl_faces.npy'
SMPL_MEAN_PARAMS_PATH = f'{ROOT_PATH}/additional/neutral_smpl_mean_params_6dpose.npz'
J_REGRESSOR_EXTRA_PATH = f'{ROOT_PATH}/additional/J_regressor_extra.npy'
COCOPLUS_REGRESSOR_PATH = f'{ROOT_PATH}/additional/cocoplus_regressor.npy'
H36M_REGRESSOR_PATH = f'{ROOT_PATH}/additional/J_regressor_h36m.npy'
VERTEX_TEXTURE_PATH = f'{ROOT_PATH}/additional/vertex_texture.npy'
CUBE_PARTS_PATH = f'{ROOT_PATH}/additional/cube_parts.npy'
UV_MAT_PATH =f'{ROOT_PATH}/additional/UV_Processed.mat'
# ------------------------ Constants ------------------------
FOCAL_LENGTH = 5000.
REGRESSOR_IMG_WH = 256
MEAN_CAM_T = [0., 0.2, 42.]
# ------------------------ SMPL Prior ------------------------
STRAP_TRAIN_PATH = '../data/strap_train.npz'
STRAP_VAL_PATH ='../data/strap_val.npz'

H36M_P1_GT = f'{ROOT_PATH}/h36m/h36m_valid_protocol1.npz'
H36M_P1_IMG_DIR = f'./data/h36mp1test/image_i0_s1.2'
H36M_P1_PR_DIR = f'./data/h36mp1test/pr_i0_s1.2'

H36M_GT = f'{ROOT_PATH}/h36m/h36m_valid_protocol2.npz'
H36M_IMG_DIR = f'./data/h36mtest/image_s1.2'
H36M_PR_DIR = f'./data/h36mtest/pr_s1.2'

D3PW_GT = f'{ROOT_PATH}/3dpw/3dpw_test.npz' #35515
D3PW_IMG_DIR = f'/strap/data/3dpwtest_new/image_i0_s1.2'
D3PW_PR_DIR = f'/strap/data/3dpwtest_new/pr_i0_s1.2'

