from scipy.io import loadmat



# data = loadmat("C:/Users/user/Desktop/proprecess1/s1/top_data.mat")
# data = data['data']
# label = loadmat("C:/Users/user/Desktop/proprecess1/s1/top_label.mat")
# label = label['top_label']
#
# RealLen = []
# Ltrial = []
# for i in range(15):
#     Ltrial.append(0)
#     RealLen.append(0)
# for item in label:
#     Ltrial[item[0]] = Ltrial[item[0]] + 1
# print(data.shape)
# print(label.shape)
# print(Ltrial)
# print(sum(Ltrial))


data = loadmat("C:/Users/user/Desktop/mean_data15_1/s1/bottom_data.mat")
data = data['bottom_data']
label = loadmat("C:/Users/user/Desktop/mean_data15_1/s1/bottom_label.mat")
label = label['label']

RealLen = []
Ltrial = []
for i in range(15):
    Ltrial.append(0)
    RealLen.append(0)
for item in label:
    Ltrial[item[0]] = Ltrial[item[0]] + 1
print(data.shape)
print(label.shape)
print(Ltrial)
print(sum(Ltrial))

data = loadmat("C:/Users/user/Desktop/mean_data15_1/s2/bottom_data.mat")
data = data['bottom_data']
label = loadmat("C:/Users/user/Desktop/mean_data15_1/s2/bottom_label.mat")
label = label['label']

RealLen = []
Ltrial = []
for i in range(15):
    Ltrial.append(0)
    RealLen.append(0)
for item in label:
    Ltrial[item[0]] = Ltrial[item[0]] + 1
print(data.shape)
print(label.shape)
print(Ltrial)
print(sum(Ltrial))