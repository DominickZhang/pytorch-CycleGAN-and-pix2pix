import numpy as np
import h5py as h5
import logging
import os
import torch
from train_swingen import create_model
from statistics_jinnian import saveImage

def pred_2_CT(array):
	return (array + 0.5)*2500.0

def cal_F1_score_volume(y_true, y_pred):
	return 2.0*np.sum(y_true*y_pred)/(np.sum(y_true) + np.sum(y_pred))

def cal_avg_dice_all_region(y_true, y_pred):
	mask_brain = y_true >= -500
	mask_air = y_true < -500
	mask_soft = (y_true > -500) * (y_true < 300)
	mask_bone = y_true > 500
	assert(np.sum(mask_brain)>0)
	assert(np.sum(mask_air)>0)
	assert(np.sum(mask_soft)>0)
	assert(np.sum(mask_bone)>0)

	pred_brain = y_pred >= -500
	pred_air = y_pred < -500
	pred_soft = (y_pred > -500) * (y_pred < 300)
	pred_bone = y_pred > 500

	dice_brain = cal_F1_score_volume(mask_brain, pred_brain)
	dice_air = cal_F1_score_volume(mask_air, pred_air)
	dice_soft = cal_F1_score_volume(mask_soft, pred_soft)
	dice_bone = cal_F1_score_volume(mask_bone, pred_bone)
	#print([dice_brain, dice_air, dice_soft, dice_bone])
	return np.mean([dice_brain, dice_air, dice_soft, dice_bone])

def plot():
	np.set_printoptions(precision=4, suppress=True)

	data_type = 'bravo'
	#filename = 'output/plot_%s_data_modified.h5'%data_type
	filename = f"/data/users/jzhang/NAS_robustness/output/plot_{data_type}_data_modified.h5"
	f = h5.File(filename, 'r')
	data_sample = f['data'][()]
	label_sample = f['label'][()]
	label_sample -= 1000
	#label_sample = CT_2_label(label_sample)
	#print(type(data_sample), type(label_sample))
	n_col = data_sample.shape[0]

	#model_name = ['bravo_advtraining_noise_epsilon_0.2_0', 'bravo_advtraining_replace_epsilon_0.2_0']
	#model_type = 'bravo_advtraining_noise_epsilon_0.2_0'
	#model_name = 'output/%s/2dunet.h5'%model_type
	#model_list = parse_models(keywords=['bravo_advtraining_noise_epsilon_0.2'])
	#model_list = ['/data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/NAS-robustness/output/bravo_regular_0/2dunet.h5', '/data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/nas-robustness/output/bravo_RobNASv2_bugfix0518_tpe117_0/2dunet.h5']
	model_list = ['/data/data_mrcv2/MCMILLAN_GROUP/50_users/jinnian/checkpoints/swingen/brats/ckpt_epoch_1.pth']
	for model_name in model_list:
		epsilon = 0.2
		#epsilon = 0.1
		logging.info(f'Loading model ({model_name})...')

		model = create_model().cuda()
		ckpt = torch.load(model_name, map_location='cpu')
		model.load_state_dict(ckpt['model'], strict=True)
		del ckpt

		inputs = torch.tensor(data_sample).permute(0,3,1,2)
		pred = model(inputs).permute(0,2,3,1).numpy()

		#x_adv_sample = FGSM(model, data_sample, CT_2_label(label_sample), batch_size=6, gradient_only=True)
		# x_adv  = data_sample + x_adv_sample*epsilon
		# pred_adv = model.predict(x_adv, batch_size=16)


		# data_range = np.max(data_sample) - np.min(data_sample)
		#label_range = np.max(label_sample) - np.min(label_sample)
		#print(np.max(label_sample), np.min(label_sample), data_range, label_range)
		#data_range = 1.64
		# psnr_x = cal_psnr_numpy(data_sample, x_adv, data_range)
		# array = np.concatenate([data_sample[np.newaxis], x_adv[np.newaxis]])

		None_row = np.array([[None for _ in range(n_col)]])
		# text = np.concatenate([None_row, psnr_x[np.newaxis]])
		# text_prefix_psnr = ['', 'PSNR: ']
		# text_label_psnr = ['Bravo Images', 'Perturbed Bravo Images with epsilon of %.2f'%epsilon]
		# saveImage(array, text, text_prefix_psnr, text_label_psnr, n_row=2, n_col=n_col, path='output/mri_%s_%.2f.png'%(model_type, epsilon))
		# #print('The shape of input array', array.shape)
		# logging.info('Image saved!')

		array = np.concatenate([data_sample[np.newaxis], label_sample[np.newaxis], pred[np.newaxis]])
		#data_range = np.max([np.max(label_sample), np.max(pred), np.max(pred_adv)]) - np.min([np.min(label_sample), np.min(pred), np.min(pred_adv)])
		pred = pred_2_CT(pred)
		# pred_adv = pred_2_CT(pred_adv)
		'''
		psnr_pred = cal_psnr_numpy(label_sample, pred_2_CT(pred), label_range)
		psnr_adv = cal_psnr_numpy(label_sample, pred_2_CT(pred_adv), label_range)
		'''
		#psnr_pred = cal_psnr_numpy(label_sample, pred, label_range)
		#psnr_adv = cal_psnr_numpy(label_sample, pred_adv, label_range)
		avg_dice_pred = []
		# avg_dice_adv = []
		for sample_index in range(len(pred)):
			#print(model_type, sample_index)
			avg_dice_pred.append(cal_avg_dice_all_region(label_sample[sample_index], pred[sample_index]))
			# avg_dice_adv.append(cal_avg_dice_all_region(label_sample[sample_index], pred_adv[sample_index]))
		avg_dice_pred = np.array(avg_dice_pred)
		# avg_dice_adv = np.array(avg_dice_adv)
		#text = np.array([[None for _ in range(5)]].append(psnr_pred).append(psnr_adv))
		text = np.concatenate([None_row, None_row, avg_dice_pred[np.newaxis]])
		#text_prefix_psnr = ['', 'PSNR: ', 'PSNR: ']
		#text_prefix_psnr = ['', 'Dice: ', 'Dice: ']
		text_prefix_psnr = ['', '', '']
		text_label_psnr = ['Input MRI Images', 'CT Images (ground truth)', 'Prediction from SwinGenerator']
		saveImage(array, text, text_prefix_psnr, text_label_psnr, n_row=3, n_col=n_col, path='output/pred_SwinGenerator.png')
		#print('The shape of input array', array.shape)
	f.close()

if __name__ == '__main__':
	plot()