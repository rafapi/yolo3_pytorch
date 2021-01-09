#!/usr/bin/python3

import os.path as path, sys, struct, inspect, functools
import urllib.request as request, numpy as np, cv2
import torch, torch.nn as nn, torchvision.ops as ops

class Scale2D(nn.Module):
	def __init__(self, n):
		super().__init__()
		self.register_parameter('alpha', torch.nn.Parameter(torch.ones([1, n, 1, 1])))
		self.register_parameter('beta', torch.nn.Parameter(torch.ones([1, n, 1, 1])))
	def forward(self, x):
		x = x * self.alpha + self.beta
		return x

class YoloConv(nn.Module):
	def __init__(self, bn, c, n, size, stride, pad, leaky_relu=True):
		super().__init__()
		self.conv = nn.Conv2d(
			in_channels=c, out_channels=n,
			kernel_size=(size, size),
			padding=(pad, pad), stride=(stride, stride),
			bias=not bn)
		if bn:
			self.bn = nn.BatchNorm2d(num_features=n, affine=True)
			self.scale = Scale2D(n)
		if leaky_relu:
			self.active = nn.LeakyReLU(negative_slope=0.1)
	def forward(self, x):
		x = self.conv(x)
		if hasattr(self, 'bn'):
			x = self.bn(x)
			x = self.scale(x)
		if hasattr(self, 'active'):
			x = self.active(x)
		return x

class YoloResBlock(nn.Module):
	def __init__(self, c, n, res):
		super().__init__()
		self.res = res
		self.yolo_conv_1 = YoloConv(True, c, n // 2, 1, 1, 0)
		self.yolo_conv_2 = YoloConv(True, n // 2, n, 3, 1, 1)
	def forward(self, x):
		y = self.yolo_conv_1(x)
		y = self.yolo_conv_2(y)
		if self.res:
			y = x + y
		return y

class Yolov3Backbone(nn.Module):
	def __init__(self, num_vals):
		super().__init__()

		seg1_mods = [] # input: data
		seg1_mods.append(YoloConv(True, 3, 32, 3, 1, 1))
		seg1_mods.append(YoloConv(True, 32, 64, 3, 2, 1))
		seg1_mods.append(YoloResBlock(64, 64, True))
		seg1_mods.append(YoloConv(True, 64, 128, 3, 2, 1))
		seg1_mods.append(YoloResBlock(128, 128, True))
		seg1_mods.append(YoloResBlock(128, 128, True))
		seg1_mods.append(YoloConv(True, 128, 256, 3, 2, 1))
		seg1_mods.append(YoloResBlock(256, 256, True))
		for i in range(7):
			seg1_mods.append(YoloResBlock(256, 256, True))

		seg2_mods = [] # seg2 input: seg1_out
		seg2_mods.append(YoloConv(True, 256, 512, 3, 2, 1))
		seg2_mods.append(YoloResBlock(512, 512, True))
		for i in range(7):
			seg2_mods.append(YoloResBlock(512, 512, True))

		seg3_mods = [] # seg3 input: seg2_out
		seg3_mods.append(YoloConv(True, 512, 1024, 3, 2, 1))
		seg3_mods.append(YoloResBlock(1024, 1024, True))
		for i in range(3):
			seg3_mods.append(YoloResBlock(1024, 1024, True))
		for i in range(2):
			seg3_mods.append(YoloResBlock(1024, 1024, False))
		seg3_mods.append(YoloConv(True, 1024, 512, 1, 1, 0))

		yolo1_mods = [] # yolo1 input: seg3_out
		yolo1_mods.append(YoloConv(True, 512, 1024, 3, 1, 1))
		yolo1_mods.append(YoloConv(False, 1024, num_vals, 1, 1, 0, False))
	
		seg4_mods = [] # seg4 input: seg2_out
		seg4_mods.append(YoloConv(True, 512, 256, 1, 1, 0))
		seg4_mods.append(nn.Upsample(scale_factor=2))

		seg5_mods = [] # seg5 input: seg4_out, seg2_out
		seg5_mods.append(YoloResBlock(768, 512, False))
		seg5_mods.append(YoloResBlock(512, 512, False))
		seg5_mods.append(YoloConv(True, 512, 256, 1, 1, 0))

		yolo2_mods = [] # yolo2 input: seg5
		yolo2_mods.append(YoloConv(True, 256, 512, 3, 1, 1))
		yolo2_mods.append(YoloConv(False, 512, num_vals, 1, 1, 0, False))

		seg6_mods = [] # seg6 input: seg5
		seg6_mods.append(YoloConv(True, 256, 128, 1, 1, 0))
		seg6_mods.append(nn.Upsample(scale_factor=2))

		yolo3_mods = [] # yolo3 input: seg6, seg1_out
		yolo3_mods.append(YoloResBlock(384, 256, False))
		for i in range(2):
			yolo3_mods.append(YoloResBlock(256, 256, False))
		yolo3_mods.append(YoloConv(False, 256, num_vals, 1, 1, 0, False))

		# DO NOT REARRANGEMENT FOLLOWING SEQUENTIALS
		# the order is matching with cfg file and weights file of darknet:
		# `yolov3-voc.cfg` & `yolov3.weights`
		self.seg1 = nn.Sequential(*seg1_mods)
		self.seg2 = nn.Sequential(*seg2_mods)
		self.seg3 = nn.Sequential(*seg3_mods)
		self.yolo1 = nn.Sequential(*yolo1_mods)
		self.seg4 = nn.Sequential(*seg4_mods)
		self.seg5 = nn.Sequential(*seg5_mods)
		self.yolo2 = nn.Sequential(*yolo2_mods)
		self.seg6 = nn.Sequential(*seg6_mods)
		self.yolo3 = nn.Sequential(*yolo3_mods)

	def forward(self, x):
		seg1_out = self.seg1.forward(x)
		seg2_out = self.seg2.forward(seg1_out)
		seg3_out = self.seg3.forward(seg2_out)
		yolo1_out = self.yolo1.forward(seg3_out)
		seg4_out = self.seg4.forward(seg3_out)
		cat42 = torch.cat((seg4_out, seg2_out), 1)
		seg5_out = self.seg5.forward(cat42)
		yolo2_out = self.yolo2.forward(seg5_out)
		seg6_out = self.seg6.forward(seg5_out)
		cat61 = torch.cat((seg6_out, seg1_out), 1)
		yolo3_out = self.yolo3.forward(cat61)
		return yolo1_out, yolo2_out, yolo3_out

class YoloLayer(nn.Module):
	def __init__(self, input_shape, img_size, anchors, masks):
		super().__init__()

		num_batch = -1 # dynamic batchsize
		num_chs = input_shape[1]
		num_rows = input_shape[2]
		num_cols = input_shape[3]
		num_ancs = len(masks)
		num_vals = num_chs // num_ancs

		self.input_shape = [num_batch, num_ancs, num_vals, num_rows, num_cols]
		self.trans_shape = [num_vals, num_batch, num_ancs, num_rows, num_cols]

		# (l[0] + [0 ~ w - 1]) * (1 / w)
		self.box_x_alpha = float(1) / num_cols
		self.box_x_beta = torch.from_numpy(np.arange(num_cols,
				dtype=np.float32)) * self.box_x_alpha
		self.box_x_beta = self.box_x_beta.reshape([1, 1, 1, 1, num_cols])

		# (l[1] + [0 ~ h - 1]) * (1 / h)
		self.box_y_alpha = float(1) / num_rows
		self.box_y_beta = torch.from_numpy(np.arange(num_rows,
				dtype=np.float32)) * self.box_y_alpha
		self.box_y_beta = self.box_y_beta.reshape([1, 1, 1, num_rows, 1])

		# l[2] * (anchor[n].w / 416)
		# l[3] * (anchor[n].h / 416)
		self.box_w_alpha = torch.zeros([num_ancs])
		self.box_h_alpha = torch.zeros([num_ancs])
		for i_mask in range(num_ancs):
			self.box_w_alpha[i_mask] = anchors[masks[i_mask]][0] \
					/ float(img_size[0])
			self.box_h_alpha[i_mask] = anchors[masks[i_mask]][1] \
					/ float(img_size[1])
		self.box_w_alpha = self.box_w_alpha.reshape([1, 1, num_ancs, 1, 1])
		self.box_h_alpha = self.box_h_alpha.reshape([1, 1, num_ancs, 1, 1])

	def forward(self, input):
		input = input.reshape(self.input_shape)
		input = input.permute(2, 0, 1, 3, 4) # v, b, a, h, w

		box_x = self.box_x_alpha * torch.sigmoid(input[0:1]) + self.box_x_beta
		box_y = self.box_y_alpha * torch.sigmoid(input[1:2]) + self.box_y_beta
		box_w = self.box_w_alpha * torch.exp(input[2:3])
		box_h = self.box_h_alpha * torch.exp(input[3:4])
		conf = torch.sigmoid(input[4:5])
		prob = torch.sigmoid(input[5:])
		if not self.training:
			prob *= conf
		output = torch.cat((box_x, box_y, box_w, box_h, conf, prob))
		return output.permute(1, 2, 3, 4, 0)

class Yolov3Model(nn.Module):
	def __init__(self, num_classes, input_shape, anchors, masks):
		super().__init__()

		# 3 anchors, 4 for boxes, 1 for confidence
		self.num_ancs = len(masks)
		self.num_vals = 4 + 1 + num_classes
		img_size = (input_shape[3], input_shape[2]) # (w, h)

		self.backbone = Yolov3Backbone(self.num_ancs * self.num_vals)
		# Inference once for determination the input shape of yolo layer
		example_out = self.backbone.forward(torch.rand(input_shape))
		out_shapes = [out.shape for out in example_out]
		yolo_layers = []
		for idx, shape in enumerate(out_shapes):
			yolo_layers.append(YoloLayer(shape,
					img_size, anchors, masks[idx]))
		self.yolo_mods = nn.ModuleList(yolo_layers)

	def forward(self, input):
		outputs = self.backbone(input)
		yolo_outs = []
		i = 0
		for yolo in self.yolo_mods:
			out = yolo(outputs[i])
			out = out.reshape(out.shape[0], -1, self.num_vals)
			yolo_outs.append(out)
			i += 1
		return torch.cat(yolo_outs, dim=1)

def load_param(mod, param_name, values, offset):
	tensor = [x[1] for x in mod.named_parameters() if x[0] == param_name][0]
	loaded = torch.tensor(values[offset:offset+tensor.numel()].copy())
	tensor.copy_(loaded.reshape_as(tensor))
	return offset + tensor.numel()

def load_params_file(filename, mod):
	f = open(filename, 'rb')
	weights_bin = f.read()
	head = struct.unpack('<3i1q', weights_bin[:20])
	values = np.frombuffer(weights_bin[20:], dtype='<f4')
	offset = 0
	with torch.no_grad():
		for mod_name, mod in model.named_modules():
			if isinstance(mod, YoloConv):
				size = mod.conv.kernel_size
				c = mod.conv.in_channels
				n = mod.conv.out_channels
				if hasattr(mod, 'bn'):
					mod.bn.reset_running_stats() # for bn.running_mean and bn.running_var
					offset = load_param(mod, 'scale.beta', values, offset)
					offset = load_param(mod, 'scale.alpha', values, offset)
					offset = load_param(mod, 'bn.bias', values, offset)
					offset = load_param(mod, 'bn.weight', values, offset)
					offset = load_param(mod, 'conv.weight', values, offset)
					gamma = 1.0 / (torch.sqrt(mod.bn.weight + 0.00001))
					beta = -mod.bn.bias * gamma
					mod.bn.weight.copy_(gamma.reshape_as(mod.bn.weight))
					mod.bn.bias.copy_(beta.reshape_as(mod.bn.bias))
				else:
					offset = load_param(mod, 'conv.bias', values, offset)
					offset = load_param(mod, 'conv.weight', values, offset)

data_path = './test/'
if len(sys.argv) > 1:
	data_path = sys.argv[1]
torch.set_printoptions(sci_mode=False)

num_classes = 80
batchsize = 1
input_size = (416, 416)
conf_thres = 0.24
nms_thres = 0.45
anchors = [(10,13),  (16,30),   (33,23),
		   (30,61),  (62,45),   (59,119),
		   (116,90), (156,198), (373,326)]
masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

# To check or download image and weight files
image_filename = path.join(data_path, 'dog.jpg')
image_file_url = 'https://github.com/pjreddie/darknet/raw/master/data/dog.jpg'
if not path.exists(image_filename):
	request.urlretrieve(image_file_url, image_filename)

weight_filename = path.join(data_path, 'yolov3.weights')
weight_file_url = 'https://pjreddie.com/media/files/yolov3.weights'
if not path.exists(weight_filename):
	request.urlretrieve(weight_file_url, weight_filename)

img = cv2.imread(image_filename, cv2.IMREAD_COLOR)
img_size = (img.shape[1], img.shape[0])
data = cv2.cvtColor(cv2.resize(img, input_size), cv2.COLOR_RGB2BGR)
data = np.transpose(data / np.float32(255), (2, 0, 1)).reshape(
		batchsize, 3, input_size[1], input_size[0])
data = torch.tensor(np.copy(data))

model = Yolov3Model(num_classes, data.shape, anchors, masks)
model.eval()
load_params_file(weight_filename, model)
batch_predict = model.forward(data)
batch_predict.reshape(batchsize, -1, 5 + num_classes)
batch_predict[:,:,5:] *= torch.gt(batch_predict[:,:,5:], conf_thres)

def nms_compare(k):
	def comp(box_a, box_b):
		if box_a[5 + k] < box_b[5 + k]: return -1
		elif box_a[5 + k] > box_b[5 + k]: return 1
		else: return 0
	return comp

def nms(all_boxes):
	for k in range(0, num_classes):
		all_boxes = sorted(all_boxes, key=functools.cmp_to_key(nms_compare(k)),
				reverse=True)
		all_boxes = torch.cat(all_boxes).reshape(-1, 85)
		for i in range(0, all_boxes.shape[0] - 1):
			if all_boxes[i][5+k] == 0:
				continue
			boxes_i = all_boxes[i:i+1,:4]
			boxes_j = all_boxes[i+1:,:4]
			boxes_iou = ops.box_iou(boxes_i, boxes_j)
			for j, iou in enumerate(boxes_iou[0]):
				if iou > nms_thres:
					all_boxes[i+1+j,5+k] = 0
	return all_boxes

for predict in batch_predict:
	all_boxes = []
	for boxes in predict:
		conf = boxes[4]
		if conf > conf_thres:
			all_boxes.append(boxes.reshape(1, -1))
	all_boxes = torch.cat(all_boxes)

	# (cx, cy, w, h) -> (x1, y1, x2, y2)
	all_boxes[:,0] -= all_boxes[:,2] / 2
	all_boxes[:,1] -= all_boxes[:,3] / 2
	all_boxes[:,2] += all_boxes[:,0]
	all_boxes[:,3] += all_boxes[:,1]
	all_boxes = nms(all_boxes)
	cls_ids = torch.argmax(all_boxes[:, 5:], dim=1)

	best_boxes = []
	for box_idx, best_cls_id in enumerate(cls_ids):
		if all_boxes[box_idx, 5 + best_cls_id] > conf_thres:
			box = torch.cat((all_boxes[box_idx, :4],
					torch.tensor([np.float32(best_cls_id)])))
			best_boxes.append(box)
	for box in best_boxes:
		pt1 = (int(box[0] * img_size[0]), int(box[1] * img_size[1]))
		pt2 = (int(box[2] * img_size[0]), int(box[3] * img_size[1]))
		cv2.rectangle(img, pt1, pt2, (0, 255, 0))

cv2.imwrite(path.join(data_path, 'predict.png'), img)

model.train()
script_model = torch.jit.script(model)
model.eval()
trace_model = torch.jit.trace(model, data)
torch.jit.save(script_model, path.join(data_path, 'yolov3.pth'))
